#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <vector>

#include "VectorVariable.hpp"

namespace {
constexpr Real tolerance = 1e-5f;
constexpr Real tolerance_loose = 1e-3f;

// Validates basic get/set operations, padding, and exception paths.
void test_value_and_set() {
    const Dim Nx = 2;
    const Dim Ny = 2;
    const Dim Nz = 2;
    VectorVariable variable(Nx, Ny, Nz, 1.0f, 1.0f, 1.0f);

    Real& stored = variable.set(0, 0, 0, 0);
    stored = 1.5f;
    assert(std::fabs(variable.value(0, 0, 0, 0) - 1.5f) < tolerance);

    variable.set(1, 1, 1, 1) = -2.0f;
    const Dim linear_index = 1 + 1 * Nx + 1 * Nx * Ny;
    assert(std::fabs(variable.value(1, linear_index) + 2.0f) < tolerance);

    // Out-of-domain coordinate should return zero due to padding
    assert(std::fabs(variable.value(0, 0, Ny, 0)) < tolerance);

    bool caught_axis_exception = false;
    try {
        (void)variable.value(-1, 0, 0, 0);
    } catch (const std::out_of_range&) {
        caught_axis_exception = true;
    }
    assert(caught_axis_exception);

    bool caught_index_exception = false;
    try {
        (void)variable.value(0, Nx * Ny * Nz);
    } catch (const std::out_of_range&) {
        caught_index_exception = true;
    }
    assert(caught_index_exception);
}

// Confirms first-derivative computations match a linear field.
void test_first_derivative() {
    const Dim Nx = 3;
    const Dim Ny = 3;
    const Dim Nz = 3;
    VectorVariable variable(Nx, Ny, Nz, 1.0f, 1.0f, 1.0f);

    for (Dim k = 0; k < Nz; ++k) {
        for (Dim j = 0; j < Ny; ++j) {
            for (Dim i = 0; i < Nx; ++i) {
                variable.set(0, i, j, k) = static_cast<Real>(i + j + k);
            }
        }
    }

    const Real derivative_x = variable.first_derivative(0, 0, 1, 1, 1);
    assert(std::fabs(derivative_x - 1.0f) < tolerance);

    const Dim index = 1 + 1 * Nx + 1 * Nx * Ny;
    const Real derivative_x_index = variable.first_derivative(0, 0, index);
    assert(std::fabs(derivative_x_index - 1.0f) < tolerance);
}

// Ensures second-derivative results are correct for a quadratic field.
void test_second_derivative() {
    const Dim Nx = 3;
    const Dim Ny = 3;
    const Dim Nz = 3;
    VectorVariable variable(Nx, Ny, Nz, 1.0f, 1.0f, 1.0f);

    for (Dim k = 0; k < Nz; ++k) {
        for (Dim j = 0; j < Ny; ++j) {
            for (Dim i = 0; i < Nx; ++i) {
                variable.set(0, i, j, k) = static_cast<Real>(i * i);
            }
        }
    }

    const Real second_derivative_x = variable.second_derivative(0, 0, 1, 1, 1);
    assert(std::fabs(second_derivative_x - 2.0f) < tolerance);

    const Dim index = 1 + 1 * Nx + 1 * Nx * Ny;
    const Real second_derivative_x_index = variable.second_derivative(0, 0, index);
    assert(std::fabs(second_derivative_x_index - 2.0f) < tolerance);
}

// Verifies divergence of a simple vector field both indexed and by coordinates.
void test_divergence() {
    const Dim Nx = 3;
    const Dim Ny = 3;
    const Dim Nz = 3;
    VectorVariable variable(Nx, Ny, Nz, 1.0f, 1.0f, 1.0f);

    for (Dim k = 0; k < Nz; ++k) {
        for (Dim j = 0; j < Ny; ++j) {
            for (Dim i = 0; i < Nx; ++i) {
                variable.set(0, i, j, k) = static_cast<Real>(i);
                variable.set(1, i, j, k) = static_cast<Real>(j);
                variable.set(2, i, j, k) = static_cast<Real>(k);
            }
        }
    }

    const Real div_point = variable.divergence(1, 1, 1);
    assert(std::fabs(div_point - 3.0f) < tolerance);

    const Dim index = 1 + 1 * Nx + 1 * Nx * Ny;
    const Real div_index = variable.divergence(index);
    assert(std::fabs(div_index - 3.0f) < tolerance);
}

// Checks construction across multiple grid sizes and shapes.
void test_constructor_various_sizes() {
    std::cout << "Testing constructor with various grid sizes...\n";
    
    // Test small grid
    VectorVariable small(1, 1, 1, 0.1f, 0.1f, 0.1f);
    assert(small.value(0, 0, 0, 0) == 0.0f);
    
    // Test medium grid
    VectorVariable medium(5, 5, 5, 0.2f, 0.2f, 0.2f);
    assert(medium.value(0, 4, 4, 4) == 0.0f);
    
    // Test rectangular grid
    VectorVariable rect(3, 7, 2, 0.5f, 0.3f, 0.4f);
    assert(rect.value(0, 2, 6, 1) == 0.0f);
    
    std::cout << "Constructor tests passed.\n";
}

// Confirms constructor handles diverse spacing values without altering defaults.
void test_constructor_various_spacings() {
    std::cout << "Testing constructor with various spacing values...\n";
    
    // Test with different dx, dy, dz values
    VectorVariable var1(2, 2, 2, 0.1f, 0.2f, 0.3f);
    VectorVariable var2(2, 2, 2, 1.0f, 2.0f, 3.0f);
    VectorVariable var3(2, 2, 2, 0.01f, 0.02f, 0.03f);
    
    // All should initialize correctly
    assert(var1.value(0, 0, 0, 0) == 0.0f);
    assert(var2.value(0, 0, 0, 0) == 0.0f);
    assert(var3.value(0, 0, 0, 0) == 0.0f);
    
    std::cout << "Spacing constructor tests passed.\n";
}

// Tests directional first derivatives for a multi-axis linear scalar field.
void test_derivatives_all_directions() {
    std::cout << "Testing derivatives in all three directions...\n";
    
    const Dim Nx = 4;
    const Dim Ny = 4;
    const Dim Nz = 4;
    VectorVariable variable(Nx, Ny, Nz, 1.0f, 1.0f, 1.0f);
    
    // Set up a function f(x,y,z) = x + 2y + 3z
    for (Dim k = 0; k < Nz; ++k) {
        for (Dim j = 0; j < Ny; ++j) {
            for (Dim i = 0; i < Nx; ++i) {
                variable.set(0, i, j, k) = static_cast<Real>(i + 2*j + 3*k);
            }
        }
    }
    
    // Test x-derivative (should be 1)
    const Real dx = variable.first_derivative(0, 0, 1, 1, 1);
    assert(std::fabs(dx - 1.0f) < tolerance);
    
    // Test y-derivative (should be 2)
    const Real dy = variable.first_derivative(0, 1, 1, 1, 1);
    assert(std::fabs(dy - 2.0f) < tolerance);
    
    // Test z-derivative (should be 3)
    const Real dz = variable.first_derivative(0, 2, 1, 1, 1);
    assert(std::fabs(dz - 3.0f) < tolerance);
    
    std::cout << "All direction derivative tests passed.\n";
}

// Evaluates derivative scaling across different grid spacings.
void test_derivatives_different_spacings() {
    std::cout << "Testing derivatives with different grid spacings...\n";
    
    const Dim Nx = 3;
    const Dim Ny = 3;
    const Dim Nz = 3;
    
    // Test with different spacing values
    VectorVariable var1(Nx, Ny, Nz, 0.5f, 1.0f, 1.0f);  // dx = 0.5
    VectorVariable var2(Nx, Ny, Nz, 1.0f, 1.0f, 1.0f);  // dx = 1.0
    VectorVariable var3(Nx, Ny, Nz, 2.0f, 1.0f, 1.0f);  // dx = 2.0
    
    // Set up f(x) = x for all variables
    for (Dim k = 0; k < Nz; ++k) {
        for (Dim j = 0; j < Ny; ++j) {
            for (Dim i = 0; i < Nx; ++i) {
                var1.set(0, i, j, k) = static_cast<Real>(i);
                var2.set(0, i, j, k) = static_cast<Real>(i);
                var3.set(0, i, j, k) = static_cast<Real>(i);
            }
        }
    }
    
    // For f(x) = x, df/dx = 1 regardless of spacing
    // The finite difference formula is (f[i+1] - f[i-1]) / (2*dx)
    // For f(x) = x: f[i+1] = i+1, f[i-1] = i-1
    // So: (i+1 - (i-1)) / (2*dx) = 2 / (2*dx) = 1/dx
    // But we want df/dx = 1, so we need to test the actual derivative value
    const Real d1 = var1.first_derivative(0, 0, 1, 1, 1);
    const Real d2 = var2.first_derivative(0, 0, 1, 1, 1);
    const Real d3 = var3.first_derivative(0, 0, 1, 1, 1);
    
    // Expected: d1 = 1/0.5 = 2, d2 = 1/1.0 = 1, d3 = 1/2.0 = 0.5
    assert(std::fabs(d1 - 2.0f) < tolerance);  // 1/0.5 = 2
    assert(std::fabs(d2 - 1.0f) < tolerance);  // 1/1.0 = 1
    assert(std::fabs(d3 - 0.5f) < tolerance);  // 1/2.0 = 0.5
    
    std::cout << "Different spacing derivative tests passed.\n";
}

// Assesses derivative behavior with zero-padding boundary conditions.
void test_derivatives_boundary_conditions() {
    std::cout << "Testing derivative boundary conditions...\n";
    
    const Dim Nx = 3;
    const Dim Ny = 3;
    const Dim Nz = 3;
    VectorVariable variable(Nx, Ny, Nz, 1.0f, 1.0f, 1.0f);
    
    // Set up a simple function
    for (Dim k = 0; k < Nz; ++k) {
        for (Dim j = 0; j < Ny; ++j) {
            for (Dim i = 0; i < Nx; ++i) {
                variable.set(0, i, j, k) = static_cast<Real>(i);
            }
        }
    }
    
    // Test derivatives at boundaries (should use padding = 0)
    // At i=0, derivative should be (1-0)/(2*1) = 0.5
    const Real d_left = variable.first_derivative(0, 0, 0, 1, 1);
    assert(std::fabs(d_left - 0.5f) < tolerance);
    
    // At i=Nx-1, derivative should be (0-1)/(2*1) = -0.5
    const Real d_right = variable.first_derivative(0, 0, Nx-1, 1, 1);
    assert(std::fabs(d_right + 0.5f) < tolerance);
    
    std::cout << "Boundary condition derivative tests passed.\n";
}

// Checks second derivatives for a separable quadratic scalar field.
void test_second_derivatives_all_directions() {
    std::cout << "Testing second derivatives in all directions...\n";
    
    const Dim Nx = 4;
    const Dim Ny = 4;
    const Dim Nz = 4;
    VectorVariable variable(Nx, Ny, Nz, 1.0f, 1.0f, 1.0f);
    
    // Set up f(x,y,z) = x^2 + 2y^2 + 3z^2
    for (Dim k = 0; k < Nz; ++k) {
        for (Dim j = 0; j < Ny; ++j) {
            for (Dim i = 0; i < Nx; ++i) {
                variable.set(0, i, j, k) = static_cast<Real>(i*i + 2*j*j + 3*k*k);
            }
        }
    }
    
    // Test x second derivative (should be 2)
    const Real d2x = variable.second_derivative(0, 0, 1, 1, 1);
    assert(std::fabs(d2x - 2.0f) < tolerance);
    
    // Test y second derivative (should be 4)
    const Real d2y = variable.second_derivative(0, 1, 1, 1, 1);
    assert(std::fabs(d2y - 4.0f) < tolerance);
    
    // Test z second derivative (should be 6)
    const Real d2z = variable.second_derivative(0, 2, 1, 1, 1);
    assert(std::fabs(d2z - 6.0f) < tolerance);
    
    std::cout << "All direction second derivative tests passed.\n";
}

// Compares numerical derivatives with analytical results on fine grids.
void test_analytical_solutions() {
    std::cout << "Testing with analytical solutions...\n";
    
    const Dim Nx = 7;  // Larger grid for better accuracy
    const Dim Ny = 7;
    const Dim Nz = 7;
    const Real dx = 0.05f;  // Smaller spacing for better accuracy
    const Real dy = 0.05f;
    const Real dz = 0.05f;
    VectorVariable variable(Nx, Ny, Nz, dx, dy, dz);
    
    // Test with f(x,y,z) = x^2 + y^2 + z^2
    // df/dx = 2x, d^2f/dx^2 = 2
    for (Dim k = 0; k < Nz; ++k) {
        for (Dim j = 0; j < Ny; ++j) {
            for (Dim i = 0; i < Nx; ++i) {
                const Real x = i * dx;
                const Real y = j * dy;
                const Real z = k * dz;
                variable.set(0, i, j, k) = x*x + y*y + z*z;
            }
        }
    }
    
    // Test at center point (i=3, j=3, k=3)
    const Real x_center = 3 * dx;
    
    const Real analytical_dx = 2.0f * x_center;
    const Real numerical_dx = variable.first_derivative(0, 0, 3, 3, 3);
    
    const Real analytical_d2x = 2.0f;
    const Real numerical_d2x = variable.second_derivative(0, 0, 3, 3, 3);
    
    // Use appropriate tolerance for finite difference approximation
    const Real tolerance_analytical = 1e-2f;  // More lenient for finite differences
    assert(std::fabs(numerical_dx - analytical_dx) < tolerance_analytical);
    assert(std::fabs(numerical_d2x - analytical_d2x) < tolerance_analytical);
    
    std::cout << "Analytical solution tests passed.\n";
}

// Exercises divergence on linear and quadratic vector fields including edges.
void test_divergence_comprehensive() {
    std::cout << "Testing comprehensive divergence calculations...\n";
    
    const Dim Nx = 3;
    const Dim Ny = 3;
    const Dim Nz = 3;
    VectorVariable variable(Nx, Ny, Nz, 1.0f, 1.0f, 1.0f);
    
    // Set up vector field: v = (x, y, z)
    // div(v) = 1 + 1 + 1 = 3
    for (Dim k = 0; k < Nz; ++k) {
        for (Dim j = 0; j < Ny; ++j) {
            for (Dim i = 0; i < Nx; ++i) {
                variable.set(0, i, j, k) = static_cast<Real>(i);  // x-component
                variable.set(1, i, j, k) = static_cast<Real>(j);  // y-component
                variable.set(2, i, j, k) = static_cast<Real>(k);  // z-component
            }
        }
    }
    
    // Test divergence at center
    const Real div_center = variable.divergence(1, 1, 1);
    assert(std::fabs(div_center - 3.0f) < tolerance);
    
    // Test divergence at corner (will be different due to padding at boundaries)
    // At corner (0,0,0), the finite difference uses padding (0) for out-of-bounds
    // For v = (x, y, z), at (0,0,0):
    // dvx/dx = (vx[1,0,0] - vx[-1,0,0]) / (2*dx) = (1 - 0) / (2*1) = 0.5
    // dvy/dy = (vy[0,1,0] - vy[0,-1,0]) / (2*dy) = (1 - 0) / (2*1) = 0.5  
    // dvz/dz = (vz[0,0,1] - vz[0,0,-1]) / (2*dz) = (1 - 0) / (2*1) = 0.5
    // So div = 0.5 + 0.5 + 0.5 = 1.5
    const Real div_corner = variable.divergence(0, 0, 0);
    assert(std::fabs(div_corner - 1.5f) < tolerance);
    
    // Test with different vector field: v = (x^2, y^2, z^2)
    // div(v) = 2x + 2y + 2z
    for (Dim k = 0; k < Nz; ++k) {
        for (Dim j = 0; j < Ny; ++j) {
            for (Dim i = 0; i < Nx; ++i) {
                variable.set(0, i, j, k) = static_cast<Real>(i*i);
                variable.set(1, i, j, k) = static_cast<Real>(j*j);
                variable.set(2, i, j, k) = static_cast<Real>(k*k);
            }
        }
    }
    
    // At (1,1,1): div(v) = 2*1 + 2*1 + 2*1 = 6
    const Real div_quadratic = variable.divergence(1, 1, 1);
    assert(std::fabs(div_quadratic - 6.0f) < tolerance);
    
    std::cout << "Comprehensive divergence tests passed.\n";
}

// Profiles repeated derivative/divergence evaluations on a large grid.
void test_performance_large_grid() {
    std::cout << "Testing performance with large grid...\n";
    
    const Dim Nx = 50;
    const Dim Ny = 50;
    const Dim Nz = 50;
    VectorVariable variable(Nx, Ny, Nz, 0.01f, 0.01f, 0.01f);
    
    // Initialize with some data
    for (Dim k = 0; k < Nz; ++k) {
        for (Dim j = 0; j < Ny; ++j) {
            for (Dim i = 0; i < Nx; ++i) {
                variable.set(0, i, j, k) = static_cast<Real>(i + j + k);
                variable.set(1, i, j, k) = static_cast<Real>(i * j + k);
                variable.set(2, i, j, k) = static_cast<Real>(i + j * k);
            }
        }
    }
    
    // Time some operations
    auto start = std::chrono::high_resolution_clock::now();
    
    // Perform many derivative calculations
    Real sum = 0.0f;
    for (int iter = 0; iter < 1000; ++iter) {
        for (Dim k = 1; k < Nz-1; ++k) {
            for (Dim j = 1; j < Ny-1; ++j) {
                for (Dim i = 1; i < Nx-1; ++i) {
                    sum += variable.first_derivative(0, 0, i, j, k);
                    sum += variable.second_derivative(0, 0, i, j, k);
                    sum += variable.divergence(i, j, k);
                }
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Large grid performance test completed in " << duration.count() << " ms\n";
    std::cout << "Sum of calculations: " << sum << " (should be non-zero)\n";
    
    // Verify the sum is reasonable (not zero, not NaN)
    assert(std::fabs(sum) > 1e-6f);
    
    std::cout << "Large grid performance tests passed.\n";
}

// Covers extreme spacings and minimal grids for numerical stability.
void test_edge_cases() {
    std::cout << "Testing edge cases...\n";
    
    // Test with minimum grid size
    VectorVariable min_grid(1, 1, 1, 1.0f, 1.0f, 1.0f);
    min_grid.set(0, 0, 0, 0) = 1.0f;
    assert(std::fabs(min_grid.value(0, 0, 0, 0) - 1.0f) < tolerance);
    
    // Test with very small spacing
    VectorVariable small_spacing(3, 3, 3, 1e-4f, 1e-4f, 1e-4f);  // Less extreme spacing
    for (Dim k = 0; k < 3; ++k) {
        for (Dim j = 0; j < 3; ++j) {
            for (Dim i = 0; i < 3; ++i) {
                small_spacing.set(0, i, j, k) = static_cast<Real>(i);
            }
        }
    }
    
    // Derivative should still work with very small spacing
    // For f(x) = x, df/dx = 1, but with spacing dx, the finite difference gives 1/dx
    const Real deriv_small = small_spacing.first_derivative(0, 0, 1, 1, 1);
    const Real expected_small = 1.0f / 1e-4f;  // 1/dx
    assert(std::fabs(deriv_small - expected_small) < tolerance_loose);
    
    // Test with very large spacing
    VectorVariable large_spacing(3, 3, 3, 1000.0f, 1000.0f, 1000.0f);
    for (Dim k = 0; k < 3; ++k) {
        for (Dim j = 0; j < 3; ++j) {
            for (Dim i = 0; i < 3; ++i) {
                large_spacing.set(0, i, j, k) = static_cast<Real>(i);
            }
        }
    }
    
    // Derivative should be much smaller with large spacing
    const Real deriv_large = large_spacing.first_derivative(0, 0, 1, 1, 1);
    assert(std::fabs(deriv_large - 1.0f/1000.0f) < tolerance_loose);
    
    std::cout << "Edge case tests passed.\n";
}

// Ensures multiple moderately large grids coexist without memory regressions.
void test_memory_usage() {
    std::cout << "Testing memory usage...\n";
    
    // Test that we can create multiple large grids without issues
    std::vector<VectorVariable> grids;
    
    for (int i = 0; i < 5; ++i) {
        grids.emplace_back(20, 20, 20, 0.1f, 0.1f, 0.1f);
        
        // Initialize with some data
        for (Dim k = 0; k < 20; ++k) {
            for (Dim j = 0; j < 20; ++j) {
                for (Dim i = 0; i < 20; ++i) {
                    grids.back().set(0, i, j, k) = static_cast<Real>(i + j + k);
                }
            }
        }
    }
    
    // Verify all grids work correctly
    for (const auto& grid : grids) {
        assert(std::fabs(grid.value(0, 10, 10, 10) - 30.0f) < tolerance);
    }
    
    std::cout << "Memory usage tests passed.\n";
}

} // namespace

int main() {
    std::cout << "========================================\n";
    std::cout << "Running comprehensive VectorVariable tests\n";
    std::cout << "========================================\n\n";
    
    // Basic tests
    test_value_and_set();
    test_first_derivative();
    test_second_derivative();
    test_divergence();
    
    // Comprehensive tests
    test_constructor_various_sizes();
    test_constructor_various_spacings();
    test_derivatives_all_directions();
    test_derivatives_different_spacings();
    test_derivatives_boundary_conditions();
    test_second_derivatives_all_directions();
    test_analytical_solutions();
    test_divergence_comprehensive();
    test_performance_large_grid();
    test_edge_cases();
    test_memory_usage();

    std::cout << "========================================\n";
    std::cout << "All VectorVariable comprehensive tests passed!\n";
    std::cout << "========================================\n";
    return 0;
}
