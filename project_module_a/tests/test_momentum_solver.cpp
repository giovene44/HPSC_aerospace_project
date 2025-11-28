#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <vector>
#include <iomanip> // Required for setprecision
#include "navier_stokes_brinkman.hpp"

int main()
{
    // 1. Setup Grid Parameters
    Real dim_x = 1.0, dim_y = 1.0, dim_z = 1.0;
    Dim Nx = 5;
    Dim Ny = 5;
    Dim Nz = 5;

    Real dx, dy, dz;
    dx = dim_x / (Nx - 0.5);
    dy = dim_y / (Ny - 0.5);
    dz = dim_z / (Nz - 0.5);
    Real dt = 0.01;

    // 2. Setup Fields
    ScalarVariable gamma_field(Nx, Ny, Nz, dx, dy, dz);
    VectorVariable rhs(Nx, Ny, Nz, dx, dy, dz);
    VectorVariable vector(Nx, Ny, Nz, dx, dy, dz);
    gamma_field.set_all(0.1f);

    // Initialize vector with test data
    for (int comp = 0; comp < 3; ++comp)
    {
        for (Dim k = 0; k < Nz; ++k)
        {
            for (Dim j = 0; j < Ny; ++j)
            {
                for (Dim i = 0; i < Nx; ++i)
                {
                    // Set boundary values to 0
                    if (i == 0 || i == Nx - 1 || j == 0 || j == Ny - 1 || k == 0 || k == Nz - 1)
                    {
                        vector.set(comp, i, j, k) = 0.0;
                    }
                    else
                    {
                        vector.set(comp, i, j, k) = static_cast<Real>(i + j + k);
                    }
                }
            }
        }
    }

    // 3. Setup Boundaries
    BoundaryFunctions u_boundary;
    std::vector<std::string> zero_bc = {"0.0", "0.0", "0.0"};
    u_boundary.set_string_expression(zero_bc);

    // 4. Instantiate Solver
    VelocitySolver velocity_solver(Nx, Ny, Nz, dx, dy, dz, dt, gamma_field, u_boundary);

    // 5. Define Strides (CORRECTION HERE)
    // We must capture Nx, Ny by value [=] so the lambda can see them.
    auto stride_x = [=](Dim j, Dim k)
    { return j * Nx + k * Nx * Ny; };
    auto stride_y = [=](Dim i, Dim k)
    { return i + k * Nx * Ny; };
    auto stride_z = [=](Dim i, Dim j)
    { return i + j * Nx; };

    // 6. Define Handlers
    DimensionsHandlerVector<decltype(stride_x)> x_vector_handler(Nx, Ny, Nz, 0, 1, 2, dx, stride_x);

    // 7. Run Test
    try
    {
        // Apply Matrix A * x -> rhs
        // Note: You need to specify the template parameters <StrideFunc, direction>
        velocity_solver.apply_matrix_operator<0, decltype(stride_x)>(vector, rhs, rhs, x_vector_handler);

        std::cout << "Matrix application successful." << std::endl;

        // B. Solve A * x_computed = rhs
        VectorVariable computed_sol(Nx, Ny, Nz, dx, dy, dz);
        computed_sol.set_all(0.0);

        // Note: Template args are <direction, StrideFunc> for block_solver
        velocity_solver.block_solver<0, decltype(stride_x)>(rhs, computed_sol, x_vector_handler);

        std::cout << "Block solver successful." << std::endl;

        // 8. Element-wise Consistency Check
        Real tolerance = 1e-6;
        bool failed = false;

        std::cout << std::fixed << std::setprecision(8);

        for (int comp = 0; comp < 3; ++comp)
        {
            for (Dim k = 0; k < Nz; ++k)
            {
                for (Dim j = 0; j < Ny; ++j)
                {
                    for (Dim i = 0; i < Nx; ++i)
                    {
                        Real original = vector.value(comp, i, j, k);
                        Real computed = computed_sol.value(comp, i, j, k);
                        Real diff = std::abs(original - computed);

                        if (diff > tolerance)
                        {
                            std::cerr << "\n[ERROR] Mismatch found!" << std::endl;
                            std::cerr << "  Index: (" << i << ", " << j << ", " << k << ") Comp: " << comp << std::endl;
                            std::cerr << "  Original: " << original << std::endl;
                            std::cerr << "  Computed: " << computed << std::endl;
                            std::cerr << "  Diff:     " << diff << std::endl;

                            failed = true;
                            // Break out of the innermost loop
                            break;
                        }
                    }
                    if (failed)
                        break;
                }
                if (failed)
                    break;
            }
            if (failed)
                break;
        }

        if (!failed)
        {
            std::cout << "\n--------------------------------------------------" << std::endl;
            std::cout << "  [PASS] Solver matrix and inversion are consistent." << std::endl;
            std::cout << "--------------------------------------------------" << std::endl;
        }
        else
        {
            std::cout << "\n--------------------------------------------------" << std::endl;
            std::cout << "  [FAIL] Test aborted due to mismatch." << std::endl;
            std::cout << "--------------------------------------------------" << std::endl;
            return -1;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Test Failed with Exception: " << e.what() << std::endl;
        return -1;
    }

    return 0;
    return 0;
}
