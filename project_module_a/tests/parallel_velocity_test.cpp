#include <iostream>
#include <cassert>
#include <cmath>
#include "Variables.hpp"
#include "DomainDecomposition.hpp"
#include "DimensionHandler.hpp"
#include "ScalarVariable.hpp"
#include "VectorVariable.hpp"
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
        std::cout << "=== Parallel Velocity Solver Test ===" << std::endl;
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
    
    // Create gamma field (uniform for testing)
    ScalarVariable gamma_field(Nx_local, Ny_local, Nz_local, dx, dy, dz);
    for (Dim i = 0; i < Nx_local; ++i) {
        for (Dim j = 0; j < Ny_local; ++j) {
            for (Dim k = 0; k < Nz_local; ++k) {
                gamma_field.set(i, j, k) = 1.0f;
            }
        }
    }
    
    // Create boundary functions (zero velocity for testing)
    BoundaryFunctions u_boundary;
    
    // Create velocity solver with decomposition
    VelocitySolver solver(Nx_local, Ny_local, Nz_local, dx, dy, dz, dt, gamma_field, u_boundary);
    solver.set_decomposition(&decomp);
    
    // Create test RHS and solution
    VectorVariable rhs(Nx_local, Ny_local, Nz_local, dx, dy, dz);
    VectorVariable solution(Nx_local, Ny_local, Nz_local, dx, dy, dz);
    
    // Initialize RHS with a simple test pattern
    for (Dim i = 0; i < Nx_local; ++i) {
        for (Dim j = 0; j < Ny_local; ++j) {
            for (Dim k = 0; k < Nz_local; ++k) {
                rhs.set(0, i, j, k) = 0.1f;
                rhs.set(1, i, j, k) = 0.1f;
                rhs.set(2, i, j, k) = 0.1f;
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
    DimensionsHandlerVector<decltype(stride_x)> dim_x(Nx_local, Ny_local, Nz_local, 0, 1, 2, dx, stride_x);
    solver.block_solver<0>(rhs, solution, dim_x);
    
    if (rank == 0) {
        std::cout << "Testing Y-direction solve..." << std::endl;
    }
    DimensionsHandlerVector<decltype(stride_y)> dim_y(Ny_local, Nx_local, Nz_local, 1, 0, 2, dy, stride_y);
    solver.block_solver<1>(rhs, solution, dim_y);
    
    if (rank == 0) {
        std::cout << "Testing Z-direction solve..." << std::endl;
    }
    DimensionsHandlerVector<decltype(stride_z)> dim_z(Nz_local, Nx_local, Ny_local, 2, 0, 1, dz, stride_z);
    solver.block_solver<2>(rhs, solution, dim_z);
    
    // Verify solution has reasonable values
    bool solution_valid = true;
    Real max_val = 0.0;
    for (int comp = 0; comp < 3; ++comp) {
        for (Dim i = 0; i < Nx_local; ++i) {
            for (Dim j = 0; j < Ny_local; ++j) {
                for (Dim k = 0; k < Nz_local; ++k) {
                    Real val = std::abs(solution.value(comp, i, j, k));
                    max_val = std::max(max_val, val);
                    if (std::isnan(val) || std::isinf(val)) {
                        solution_valid = false;
                    }
                }
            }
        }
    }
    
    int local_result = solution_valid ? 1 : 0;
    int global_result = 0;
    MPI_Reduce(&local_result, &global_result, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    
    Real global_max = 0.0;
    MPI_Reduce(&max_val, &global_max, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "Maximum solution value: " << global_max << std::endl;
        if (global_result) {
            std::cout << "\n=== Parallel Velocity Solver Test PASSED ===" << std::endl;
        } else {
            std::cout << "\n=== Parallel Velocity Solver Test FAILED ===" << std::endl;
        }
    }
    
    MPI_Finalize();
    return (rank == 0 && !global_result) ? 1 : 0;
#else
    std::cout << "This test requires MPI. Please compile with USE_MPI=1" << std::endl;
    return 1;
#endif
}
