#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include "SchurComplementSolver.hpp"
#include "MPICommunicator.hpp"

int main(int argc, char** argv) {
    // Initialize MPI
    MPICommunicator comm;
    comm.init(&argc, &argv);

    int rank = 0;
    int size = 1;
#ifdef USE_MPI
    rank = comm.get_rank();
    size = comm.get_size();
#endif

    // Default problem size and iteration count
    int N = 100;
    int num_iterations = 100;

    if (argc > 1) {
        N = std::atoi(argv[1]);
    }
    if (argc > 2) {
        num_iterations = std::atoi(argv[2]);
    }

    // Setup global tridiagonal system coefficients
    std::vector<Real> a_global(N, -1.0);
    std::vector<Real> b_global(N, 2.0);
    std::vector<Real> c_global(N, -1.0);
    std::vector<Real> rhs_global(N);

    // Boundary conditions
    a_global[0] = 0.0;
    c_global[N-1] = 0.0;

    // RHS: For -u'' = sin(πx), we need h² * f on the right-hand side
    const Real pi = 3.14159265358979323846;
    Real h = 1.0 / (N - 1);
    Real h2 = h * h;
    for (int i = 0; i < N; ++i) {
        Real x = i * h;
        rhs_global[i] = h2 * std::sin(pi * x);
    }

    // Create Schur solver - determines local partition
    SchurComplementSolver schur(N, size, rank, comm);
    int local_N = schur.get_local_N();
    int global_start = schur.get_global_start();

    // Extract local portion of coefficients
    std::vector<Real> a_local(local_N);
    std::vector<Real> b_local(local_N);
    std::vector<Real> c_local(local_N);
    std::vector<Real> rhs_local(local_N);

    for (int i = 0; i < local_N; ++i) {
        int global_i = global_start + i;

        if (global_i >= 0 && global_i < N) {
            a_local[i] = a_global[global_i];
            b_local[i] = b_global[global_i];
            c_local[i] = c_global[global_i];
            rhs_local[i] = rhs_global[global_i];
        }
    }

    // Preprocess ONCE (outside timing) - this is amortized over many solves
    schur.preprocess(a_local, b_local, c_local);

    // Synchronize before timing
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    // Time multiple solve iterations (typical time-stepping scenario)
    std::vector<Real> x_local;
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < num_iterations; ++iter) {
        schur.solve(rhs_local, x_local);
    }
    auto end = std::chrono::high_resolution_clock::now();

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    // Only rank 0 prints the timing (average per solve)
    if (rank == 0) {
        std::chrono::duration<double> elapsed = end - start;
        double avg_time = elapsed.count() / num_iterations;
        std::cout << std::fixed << std::setprecision(6) << avg_time << std::endl;
    }

    comm.finalize();
    return 0;
}
