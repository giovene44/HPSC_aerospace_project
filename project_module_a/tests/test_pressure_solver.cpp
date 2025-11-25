#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <vector>
#include <iomanip>

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

    // 2. Setup Fields (ScalarVariables for Pressure)
    ScalarVariable p_exact(Nx, Ny, Nz, dx, dy, dz);
    ScalarVariable p_rhs(Nx, Ny, Nz, dx, dy, dz);
    ScalarVariable p_computed(Nx, Ny, Nz, dx, dy, dz);

    // Initialize exact pressure with test data
    for (Dim k = 0; k < Nz; ++k)
    {
        for (Dim j = 0; j < Ny; ++j)
        {
            for (Dim i = 0; i < Nx; ++i)
            {
                // Fill with arbitrary values
                p_exact.set(i, j, k) = static_cast<Real>(i + j + k + 1.0);
            }
        }
    }

    // 3. Setup Boundaries
    // Pressure boundary conditions are typically 1 component ("0.0")
    BoundaryFunctions p_boundary;
    std::vector<std::string> zero_bc = {"0.0", "0.0", "0.0"};
    p_boundary.set_string_expression(zero_bc);

    // 4. Instantiate Solver
    PressureSolver pressure_solver(Nx, Ny, Nz, dx, dy, dz, dt, p_boundary);

    // 5. Define Strides
    // We use [=] to capture Nx, Ny, Nz by value
    auto stride_x = [=](Dim j, Dim k)
    { return j * Nx + k * Nx * Ny; };

    // 6. Define Handler
    DimensionsHandlerScalar<decltype(stride_x)> x_scalar_handler(Nx, Ny, Nz, dx, stride_x);

    // 7. Run Test
    try
    {
        std::cout << "--- Testing Pressure Solver Consistency (X-Direction) ---" << std::endl;

        // A. Apply Matrix A * x -> rhs
        // Template args: <direction, StrideFunc> (Order may vary based on your class definition, checking previous context...)
        // Previous context defined: template <Dim direction, typename StrideFunc> for PressureSolver functions.
        pressure_solver.apply_matrix_operator<0, decltype(stride_x)>(p_exact, p_rhs, x_scalar_handler);

        std::cout << "Matrix application successful." << std::endl;

        // B. Solve A * x_computed = rhs
        p_computed.set_all(0.0);

        pressure_solver.block_solver<0, decltype(stride_x)>(p_rhs, p_computed, x_scalar_handler);

        std::cout << "Block solver successful." << std::endl;

        // 8. Element-wise Consistency Check
        Real tolerance = 1e-4;
        bool failed = false;

        std::cout << std::fixed << std::setprecision(8);

        for (Dim k = 0; k < Nz; ++k)
        {
            for (Dim j = 0; j < Ny; ++j)
            {
                for (Dim i = 0; i < Nx; ++i)
                {
                    Real original = p_exact.get(i, j, k);
                    Real computed = p_computed.get(i, j, k);
                    Real diff = std::abs(original - computed);

                    if (diff > tolerance)
                    {
                        std::cerr << "\n[ERROR] Mismatch found!" << std::endl;
                        std::cerr << "  Index: (" << i << ", " << j << ", " << k << ")" << std::endl;
                        std::cerr << "  Original: " << original << std::endl;
                        std::cerr << "  Computed: " << computed << std::endl;
                        std::cerr << "  Diff:     " << diff << std::endl;

                        failed = true;
                        break; // Break innermost
                    }
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
            std::cout << "  [PASS] Pressure Solver matrix and inversion are consistent." << std::endl;
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
}