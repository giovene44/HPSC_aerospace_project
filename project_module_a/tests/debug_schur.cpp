#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "SchurComplementSolver.hpp"
#include "MPICommunicator.hpp"

int main(int argc, char** argv) {
    MPICommunicator comm;
    comm.init(&argc, &argv);

    int rank = 0;
    int size = 1;
#ifdef USE_MPI
    rank = comm.get_rank();
    size = comm.get_size();
#endif

    int N = 20;  // Small size for debugging
    if (argc > 1) N = std::atoi(argv[1]);

    // Setup global system
    std::vector<Real> a_global(N, -1.0);
    std::vector<Real> b_global(N, 2.0);
    std::vector<Real> c_global(N, -1.0);
    std::vector<Real> rhs_global(N);

    a_global[0] = 0.0;
    c_global[N-1] = 0.0;

    const Real pi = 3.14159265358979323846;
    Real h = 1.0 / (N - 1);
    Real h2 = h * h;
    for (int i = 0; i < N; ++i) {
        Real x = i * h;
        rhs_global[i] = h2 * std::sin(pi * x);
    }

    SchurComplementSolver schur(N, size, rank, comm);
    int local_N = schur.get_local_N();
    int global_start = schur.get_global_start();

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

    // Print local setup
    std::cout << "Rank " << rank << ": local_N=" << local_N
              << " global_start=" << global_start
              << " n_internal=" << schur.get_n_internal()
              << " has_left=" << schur.get_schur_data().has_left_interface
              << " has_right=" << schur.get_schur_data().has_right_interface
              << std::endl;

    std::cout << "Rank " << rank << ": rhs_local = [";
    for (int i = 0; i < local_N; ++i) {
        std::cout << rhs_local[i];
        if (i < local_N - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    schur.preprocess(a_local, b_local, c_local);

    // Print Schur matrix on rank 0
    if (rank == 0 && size > 1) {
        const auto& data = schur.get_schur_data();
        int num_iface = data.get_num_interfaces();
        std::cout << "Schur matrix (size " << num_iface << "):" << std::endl;
        std::cout << "  schur_a = [";
        for (int i = 0; i < num_iface; ++i) std::cout << data.schur_a[i] << " ";
        std::cout << "]" << std::endl;
        std::cout << "  schur_b = [";
        for (int i = 0; i < num_iface; ++i) std::cout << data.schur_b[i] << " ";
        std::cout << "]" << std::endl;
        std::cout << "  schur_c = [";
        for (int i = 0; i < num_iface; ++i) std::cout << data.schur_c[i] << " ";
        std::cout << "]" << std::endl;
    }

    // Print block data for each rank
    {
        const auto& d = schur.get_schur_data();
        const auto& blk = d.block_data;
        std::cout << "Rank " << rank << " block_data:"
                  << " b_L=" << blk.b_left_interface
                  << " b_R=" << blk.b_right_interface
                  << " sD_L=" << blk.schur_diag_left
                  << " sD_R=" << blk.schur_diag_right
                  << " c_L=" << blk.c_left_interface
                  << " a_R=" << blk.a_right_interface
                  << std::endl << std::flush;
    }

    std::vector<Real> x_local;
    std::cout << "Rank " << rank << " calling solve..." << std::endl << std::flush;
    schur.solve(rhs_local, x_local);
    std::cout << "Rank " << rank << " solve done." << std::endl << std::flush;

    // Print interface solution
    {
        const auto& d = schur.get_schur_data();
        std::cout << "Rank " << rank << ": interface_solution = [";
        for (size_t i = 0; i < d.interface_solution.size(); ++i) {
            std::cout << d.interface_solution[i];
            if (i < d.interface_solution.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    std::cout << "Rank " << rank << ": x_local = [";
    for (int i = 0; i < local_N; ++i) {
        std::cout << x_local[i];
        if (i < local_N - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    comm.finalize();
    return 0;
}
