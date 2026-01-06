// Detailed debug for np=4 interface assembly
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "SchurComplementSolver.hpp"
#include "MPICommunicator.hpp"

int main(int argc, char** argv) {
    MPICommunicator comm;
    comm.init(&argc, &argv);

    int rank = comm.get_rank();
    int size = comm.get_size();

    if (size != 4) {
        if (rank == 0) {
            std::cout << "This debug program requires exactly 4 processes!" << std::endl;
        }
        comm.finalize();
        return 1;
    }

    int N = 100;

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

    // Create solver
    SchurComplementSolver schur(N, size, rank, comm);
    int local_N = schur.get_local_N();
    int global_start = schur.get_global_start();

    // Extract local coefficients
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

    // Print domain info
    std::cout << "Rank " << rank << ": local_N=" << local_N
              << ", global_start=" << global_start
              << ", owns global[" << global_start << ".." << (global_start + local_N - 1) << "]"
              << std::endl;

    // Preprocess
    schur.preprocess(a_local, b_local, c_local);

    // Get interface indices
    int left_iface_idx = schur.get_interface_index(0);
    int right_iface_idx = schur.get_interface_index(1);

    std::cout << "Rank " << rank << ": left_iface_idx=" << left_iface_idx
              << ", right_iface_idx=" << right_iface_idx << std::endl;

    // Get Schur contributions
    const SchurComplementSolver::LocalBlock& block = schur.get_block_data();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Rank " << rank << " Schur contributions:" << std::endl;
    std::cout << "  schur_diag_left = " << block.schur_diag_left << std::endl;
    std::cout << "  schur_diag_right = " << block.schur_diag_right << std::endl;
    std::cout << "  schur_offdiag_to_left = " << block.schur_offdiag_to_left << std::endl;
    std::cout << "  schur_offdiag_to_right = " << block.schur_offdiag_to_right << std::endl;

    // Solve to trigger assembly
    std::vector<Real> x_local;
    schur.solve(rhs_local, x_local);

    // Print assembled Schur system (only rank 0)
    if (rank == 0) {
        std::cout << "\n=== Assembled Schur System (3 interfaces) ===" << std::endl;
        const auto& schur_a = schur.get_schur_a();
        const auto& schur_b = schur.get_schur_b();
        const auto& schur_c = schur.get_schur_c();
        const auto& schur_rhs = schur.get_schur_rhs();

        for (int i = 0; i < 3; ++i) {
            std::cout << "Interface " << i << " (between ranks " << i << " and " << (i+1) << "):" << std::endl;
            std::cout << "  a[" << i << "] = " << schur_a[i]
                      << ", b[" << i << "] = " << schur_b[i]
                      << ", c[" << i << "] = " << schur_c[i]
                      << ", rhs[" << i << "] = " << schur_rhs[i] << std::endl;
        }

        std::cout << "\nInterface solution:" << std::endl;
        const auto& iface_sol = schur.get_interface_solution();
        for (int i = 0; i < 3; ++i) {
            std::cout << "  u[interface " << i << "] = " << iface_sol[i] << std::endl;
        }
    }

    comm.finalize();
    return 0;
}
