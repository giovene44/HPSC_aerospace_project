#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
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

    // Default problem size
    int N = 100;

    if (argc > 1) {
        N = std::atoi(argv[1]);
    }

    if (rank == 0) {
        std::cout << "Parallel Schur Complement Solver" << std::endl;
        std::cout << "Problem size: N = " << N << std::endl;
        std::cout << "MPI processes: " << size << std::endl;
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

    std::cout << "Rank " << rank << ": local_N = " << local_N
              << ", global_start = " << global_start << std::endl;

    // Extract local portion of coefficients
    std::vector<Real> a_local(local_N);
    std::vector<Real> b_local(local_N);
    std::vector<Real> c_local(local_N);
    std::vector<Real> rhs_local(local_N);

    for (int i = 0; i < local_N; ++i) {
        // global_start already accounts for interface overlap
        int global_i = global_start + i;

        if (global_i >= 0 && global_i < N) {
            a_local[i] = a_global[global_i];
            b_local[i] = b_global[global_i];
            c_local[i] = c_global[global_i];
            rhs_local[i] = rhs_global[global_i];
        }
    }

    // Solve
    schur.preprocess(a_local, b_local, c_local);
    std::vector<Real> x_local;
    schur.solve(rhs_local, x_local);

    // Gather solution to rank 0
    std::vector<Real> x_global;
    if (rank == 0) {
        x_global.resize(N);
        // Copy rank 0's portion
        for (int i = 0; i < local_N; ++i) {
            x_global[i] = x_local[i];
        }

#ifdef USE_MPI
        // Receive from other ranks
        int next_idx = local_N;
        for (int r = 1; r < size; ++r) {
            int recv_count;
            MPI_Recv(&recv_count, 1, MPI_INT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::vector<Real> recv_buf(recv_count);
            MPI_Recv(recv_buf.data(), recv_count, MPI_FLOAT, r, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = 0; i < recv_count && next_idx < N; ++i) {
                x_global[next_idx++] = recv_buf[i];
            }
        }
#endif
    } else {
#ifdef USE_MPI
        // Send internal points (skip left interface to avoid duplication)
        int send_count = local_N - 1;
        MPI_Send(&send_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&x_local[1], send_count, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
#endif
    }

    // Save solution (only rank 0)
    if (rank == 0) {
        std::string filename = "solution_parallel_N" + std::to_string(N) +
                               "_np" + std::to_string(size) + ".dat";
        std::ofstream out(filename);
        out.precision(16);

        for (int i = 0; i < N; ++i) {
            out << i << " " << x_global[i] << "\n";
        }
        out.close();

        std::cout << "Solution saved to: " << filename << std::endl;
        std::cout << "First 5 values: ";
        for (int i = 0; i < std::min(5, N); ++i) {
            std::cout << x_global[i] << " ";
        }
        std::cout << std::endl;
    }

    comm.finalize();
    return 0;
}
