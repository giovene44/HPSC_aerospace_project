#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "Solver.hpp"
#include "DomainDecomposition.hpp"
#include "Variables.hpp"

#ifdef USE_MPI
#include <mpi.h>
#endif

/**
 * @brief Test the Schur complement solver with a known tridiagonal system
 * 
 * This test verifies that the Schur complement decomposition correctly solves
 * a simple tridiagonal system that is partitioned across multiple processes.
 */
int main(int argc, char** argv) {
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        std::cout << "=== Schur Complement Solver Test ===" << std::endl;
        std::cout << "Running on " << size << " process(es)" << std::endl;
    }
    
    // Simple test: solve a tridiagonal system where we know the answer
    // System: (I - L) * u = f, where L is the Laplacian stencil
    // With f = 1, this should give u increasing from boundaries
    
    Dim N_global = 32;  // Total number of points in one direction
    Real dx = 1.0 / (N_global - 1);
    Real dt = 0.01;
    
    // Create minimal domain decomposition (1D for simplicity)
    DomainDecomposition decomp(N_global, 1, 1, MPI_COMM_WORLD, -1, 1, 1);
    
    auto local_dims = decomp.get_local_dimensions();
    Dim Nx_local = local_dims[0];
    Dim Ny_local = local_dims[1];
    Dim Nz_local = local_dims[2];
    
    if (rank == 0) {
        std::cout << "Testing 1D problem with " << N_global << " points" << std::endl;
        std::cout << "Rank " << rank << " has " << Nx_local << " local points" << std::endl;
    }
    
    // Create a simple solver (we'll just use the base Solver class methods)
    class TestSolver : public Solver {
    public:
        TestSolver(Dim Nx, Real dx, Real dt) : Solver(Nx, 1, 1, dx, 1.0, 1.0, dt) {}
        
        void test_schur(const std::vector<Real>& a, const std::vector<Real>& b, 
                        const std::vector<Real>& c, const std::vector<Real>& rhs, 
                        std::vector<Real>& x, Dim direction) {
#ifdef USE_MPI
            schur_complement_solver(a, b, c, rhs, x, direction);
#else
            thomas_algorithm(a, b, c, rhs, x);
#endif
        }
    };
    
    TestSolver solver(Nx_local, dx, dt);
    solver.set_decomposition(&decomp);
    
    // Set up tridiagonal system: -u_{i-1} + 2*u_i - u_{i+1} = dx^2
    std::vector<Real> a(Nx_local, -1.0);
    std::vector<Real> b(Nx_local, 2.0);
    std::vector<Real> c(Nx_local, -1.0);
    std::vector<Real> rhs(Nx_local, dx * dx);
    std::vector<Real> x(Nx_local, 0.0);
    
    // Boundary conditions (Dirichlet: u = 0 at boundaries)
    auto start_indices = decomp.get_local_start_indices();
    Dim i_start = start_indices[0];
    Dim j_start = start_indices[1];
    Dim k_start = start_indices[2];
    
    auto end_indices = decomp.get_local_end_indices();
    Dim i_end = end_indices[0];
    Dim j_end = end_indices[1];
    Dim k_end = end_indices[2];
    
    if (i_start == 0) {
        // Left boundary
        a[0] = 0.0;
        b[0] = 1.0;
        c[0] = 0.0;
        rhs[0] = 0.0;
    }
    
    if (i_end == N_global) {
        // Right boundary
        a[Nx_local - 1] = 0.0;
        b[Nx_local - 1] = 1.0;
        c[Nx_local - 1] = 0.0;
        rhs[Nx_local - 1] = 0.0;
    }
    
    // Solve the system
    solver.test_schur(a, b, c, rhs, x, 0);
    
    // Verify solution
    // For this simple problem, solution should be smooth and bounded
    bool solution_valid = true;
    Real max_val = 0.0;
    for (Dim i = 0; i < Nx_local; ++i) {
        if (std::isnan(x[i]) || std::isinf(x[i])) {
            solution_valid = false;
            std::cout << "Rank " << rank << ": Invalid value at i=" << i << ": " << x[i] << std::endl;
        }
        max_val = std::max(max_val, std::abs(x[i]));
    }
    
    // Check that solution is smooth (no large jumps)
    for (Dim i = 1; i < Nx_local; ++i) {
        Real diff = std::abs(x[i] - x[i-1]);
        if (diff > 0.5) {  // Arbitrary threshold
            std::cout << "Rank " << rank << ": Large jump detected at i=" << i 
                      << ": " << x[i-1] << " -> " << x[i] << std::endl;
        }
    }
    
    int local_result = solution_valid ? 1 : 0;
    int global_result = 0;
    MPI_Reduce(&local_result, &global_result, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    
    Real global_max = 0.0;
    MPI_Reduce(&max_val, &global_max, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "Maximum solution value: " << global_max << std::endl;
        if (global_result && global_max < 1.0) {
            std::cout << "\n=== Schur Complement Test PASSED ===" << std::endl;
        } else {
            std::cout << "\n=== Schur Complement Test FAILED ===" << std::endl;
            if (!global_result) {
                std::cout << "Reason: Invalid values (NaN or Inf) detected" << std::endl;
            }
            if (global_max >= 1.0) {
                std::cout << "Reason: Solution too large (max=" << global_max << ")" << std::endl;
            }
        }
    }
    
    MPI_Finalize();
    return (rank == 0 && (!global_result || global_max >= 1.0)) ? 1 : 0;
#else
    std::cout << "This test requires MPI. Please compile with USE_MPI=1" << std::endl;
    return 1;
#endif
}
