#include <iostream>
#include <cassert>
#include <cmath>
#include "Variables.hpp"
#include "DomainDecomposition.hpp"
#include "DimensionHandler.hpp"
#include "ScalarVariable.hpp"
#include "BoundaryFunctions.hpp"
#include "Solver.hpp"

#ifdef USE_MPI
#include <mpi.h>
#endif

int main(int argc, char** argv) {
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        std::cout << "=== Parallel Pressure Solver Test ===" << std::endl;
        std::cout << "Running on " << size << " process(es)" << std::endl;
    }
    
    // Test configuration
    Dim Nx = 16, Ny = 16, Nz = 16;
    Real dx = 1.0 / (Nx - 0.5);
    Real dy = 1.0 / (Ny - 0.5);
    Real dz = 1.0 / (Nz - 0.5);
    Real dt = 0.01;
    
    // Create domain decomposition
    DomainDecomposition decomp(Nx, Ny, Nz, MPI_COMM_WORLD, -1, -1, -1);
    
    // Get local dimensions (use explicit variables to avoid C++20 structured binding capture issue)
    auto local_dims = decomp.get_local_dimensions();
    Dim Nx_local = local_dims[0];
    Dim Ny_local = local_dims[1];
    Dim Nz_local = local_dims[2];
    
    // Create boundary functions (homogeneous Neumann for pressure)
    BoundaryFunctions p_boundary;
    
    // Create pressure solver with decomposition
    PressureSolver solver(Nx_local, Ny_local, Nz_local, dx, dy, dz, dt, p_boundary);
    solver.set_decomposition(&decomp);
    
    // Create test RHS (simple constant field)
    ScalarVariable rhs(Nx_local, Ny_local, Nz_local, dx, dy, dz);
    ScalarVariable solution(Nx_local, Ny_local, Nz_local, dx, dy, dz);
    
    // Initialize RHS with a simple test pattern
    for (Dim i = 0; i < Nx_local; ++i) {
        for (Dim j = 0; j < Ny_local; ++j) {
            for (Dim k = 0; k < Nz_local; ++k) {
                rhs.set(i, j, k) = 1.0f;
            }
        }
    }
    
    // Define stride functions
    auto stride_x = [=](Dim j, Dim k) { return j * Nx_local + k * Nx_local * Ny_local; };
    auto stride_y = [=](Dim i, Dim k) { return i + k * Nx_local * Ny_local; };
    auto stride_z = [=](Dim i, Dim j) { return i + j * Nx_local; };
    
    // Solve in each direction
    if (rank == 0) {
        std::cout << "Testing X-direction solve..." << std::endl;
    }
    DimensionsHandlerScalar<decltype(stride_x)> dim_x(Nx_local, Ny_local, Nz_local, dx, stride_x);
    solver.block_solver<0>(rhs, solution, dim_x);
    
    if (rank == 0) {
        std::cout << "Testing Y-direction solve..." << std::endl;
    }
    DimensionsHandlerScalar<decltype(stride_y)> dim_y(Ny_local, Nx_local, Nz_local, dy, stride_y);
    solver.block_solver<1>(rhs, solution, dim_y);
    
    if (rank == 0) {
        std::cout << "Testing Z-direction solve..." << std::endl;
    }
    DimensionsHandlerScalar<decltype(stride_z)> dim_z(Nz_local, Nx_local, Ny_local, dz, stride_z);
    solver.block_solver<2>(rhs, solution, dim_z);
    
    // Verify solution is reasonable (should be positive)
    bool all_positive = true;
    for (Dim i = 0; i < Nx_local; ++i) {
        for (Dim j = 0; j < Ny_local; ++j) {
            for (Dim k = 0; k < Nz_local; ++k) {
                if (solution.get(i, j, k) < 0) {
                    all_positive = false;
                }
            }
        }
    }
    
    int local_result = all_positive ? 1 : 0;
    int global_result = 0;
    MPI_Reduce(&local_result, &global_result, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        if (global_result) {
            std::cout << "\n=== Parallel Pressure Solver Test PASSED ===" << std::endl;
        } else {
            std::cout << "\n=== Parallel Pressure Solver Test FAILED ===" << std::endl;
        }
    }
    
    MPI_Finalize();
    return (rank == 0 && !global_result) ? 1 : 0;
#else
    std::cout << "This test requires MPI. Please compile with USE_MPI=1" << std::endl;
    return 1;
#endif
}
