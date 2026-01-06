#include "VectorVariable.hpp"
#include "BoundaryFunctions.hpp"
#include "Variables.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>

Real exact_first_derivative(Real x, Real y, Real z, Real t, Dim component, Dim direction)
{
    // Example: u = x^2, v = y^2, w = z^2
    // du/dx = 2x, dv/dy = 2y, dw/dz = 2z
    if (component == 0 && direction == 0)
        return cos(x);
    if (component == 1 && direction == 1)
        return cos(y);
    if (component == 2 && direction == 2)
        return cos(z);
    return 0.0;
}

Real exact_second_derivative(Real x, Real y, Real z, Real t, Dim component, Dim direction)
{
    // Example: u = x^2, v = y^2, w = z^2
    // d2u/dx2 = 2, d2v/dy2 = 2, d2w/dz2 = 2
    if (component == 0 && direction == 0)
        return -sin(x);
    if (component == 1 && direction == 1)
        return -sin(y);
    if (component == 2 && direction == 2)
        return -sin(z);
    return 0.0;
}

int main()
{
    std::vector<std::string> f_values = {"sin(x)", "sin(y)", "sin(z)"};
    BoundaryFunctions boundary_functions;
    boundary_functions.set_string_expression(f_values);

    // Test 1: Construction and size
    // VectorVariable vec(10, 10, 10, 0.1, 0.1, 0.1);
    // assert(vec.size() == 3);
    // assert(vec.elements_per_component() == 1000);
    // assert(vec.get_Nx() == 10 && vec.get_Ny() == 10 && vec.get_Nz() == 10);

    // // Test 2: set_all and value access
    // vec.set_all(5.0);
    // assert(vec.value(0, 5, 5, 5) == 5.0);
    // assert(vec.value(1, 3, 3, 3) == 5.0);
    // assert(vec.value(2, 0, 0, 0) == 5.0);

    // // Test 3: Individual element setting and getting
    // vec.set(0, 2, 3, 4) = 42.0;
    // assert(vec.value(0, 2, 3, 4) == 42.0);

    // // Test 4: Boundary value handling
    // assert(vec.value(0, -1, 5, 5) == 0.0);
    // assert(vec.value(1, 5, 100, 5) == 0.0);
    // assert(vec.value(2, 5, 5, -1) == 0.0);

    // // Test 5: Out of range exceptions
    // try
    // {
    //     vec.value(5, 5, 5, 5);
    //     assert(false);
    // }
    // catch (const std::out_of_range &)
    // {
    // }

    // // Test 6: First derivative (interior points)
    // VectorVariable vec2(10, 10, 10, 1.0, 1.0, 1.0);
    // vec2.set_all(boundary_functions, 0.0, false);
    // Real deriv_x = vec2.first_derivative(0, 0, 5, 5, 5, boundary_functions, 0.0);
    // assert(std::abs(deriv_x - exact_first_derivative(5.0, 5.0, 5.0, 0.0, 0, 0)) < 1e-6);

    // // Test 7: Second derivative
    // Real second_deriv = vec2.second_derivative(0, 0, 5, 5, 5);
    // assert(std::abs(second_deriv - exact_second_derivative(5.0, 5.0, 5.0, 0.0, 0, 0)) < 1e-6);
    // // Test 8: Divergence
    // Real div = vec2.divergence(5, 5, 5, boundary_functions, 0.0);
    // assert(std::abs(div - (exact_first_derivative(5.0, 5.0, 5.0, 0.0, 0, 0) + exact_first_derivative(5.0, 5.0, 5.0, 0.0, 1, 1) + exact_first_derivative(5.0, 5.0, 5.0, 0.0, 2, 2))) < 1e-6);

    // // Test 9: Vector addition
    // VectorVariable vec3(10, 10, 10, 1.0, 1.0, 1.0);
    // vec3.set_all(3.0);
    // VectorVariable vec4 = vec2 + vec3;
    // assert(vec4.value(0, 5, 5, 5) == 28.0);

    // // Test 10: Vector subtraction
    // VectorVariable vec5 = vec2 - vec3;
    // assert(vec5.value(0, 5, 5, 5) == 22.0);

    // // Test 11: Compound assignment
    // vec3 += vec2;
    // assert(vec3.value(0, 5, 5, 5) == 28.0);

    // // Test 12: Named accessors
    // vec.set_all(7.0);
    // assert(vec.x().get(2, 2, 2) == 7.0);
    // assert(vec.y().get(2, 2, 2) == 7.0);
    // assert(vec.z().get(2, 2, 2) == 7.0);

    Dim Nx1 = 10, Ny1 = 10, Nz1 = 10;
    Real L = 2.0*M_PI;
    std::vector<Real> error_first, error_second;

    // Test 13: Convergence study for first and second derivatives
    std::cout << "\nConvergence Study (L2 Norm):\n";
    std::cout << "Grid\t\tFirst Deriv L2 Error\tSecond Deriv L2 Error\n";

    for (int run = 0; run < 4; run++)
    {
        Dim Nx = Nx1 * std::pow(2, run);
        Dim Ny = Ny1 * std::pow(2, run);
        Dim Nz = Nz1 * std::pow(2, run);
        Real dx = L / Real(Nx - 0.5);
        Real dy = L / Real(Ny - 0.5);
        Real dz = L / Real(Nz - 0.5);

        VectorVariable vec(Nx, Ny, Nz, dx, dy, dz);

        vec.set_all(boundary_functions, 0.0, true);

        Real l2_error_first = 0.0;
        Real l2_error_second = 0.0;

        for (int i = 1; i < Nx - 1; ++i)
        {
            for (int j = 1; j < Ny - 1; ++j)
            {
                for (int k = 1; k < Nz - 1; ++k)
                {
                    for (int comp = 0; comp < 3; comp++)
                    {
                        Real x = i * dx + (comp == 0 ? 0.5 * dx : 0.0);
                        Real y = j * dy + (comp == 1 ? 0.5 * dy : 0.0);
                        Real z = k * dz + (comp == 2 ? 0.5 * dz : 0.0);

                        Real exact_first = exact_first_derivative(x, y, z, Real(0.0),comp, comp);
                        Real exact_second = exact_second_derivative(x, y, z, Real(0.0),comp, comp);

                        Real numerical_first = vec.first_derivative(comp, comp, i, j, k, boundary_functions, 0.0);
                        Real numerical_second = vec.second_derivative(comp, comp, i, j, k);

                        if (comp == 0)
                        {
                            if(std::abs(exact_first - numerical_first) > 1e-4)
                            {
                                std::cout << "Mismatch detected!" << std::endl;
                                std::cout << std::scientific << std::setprecision(15);
                                std::cout << "x=" << x << ", y=" << y << ", z=" << z << std::endl;
                                std::cout << "i=" << i << ", j=" << j << ", k=" << k << ", comp=" << comp << std::endl;
                                std::cout << "f_x_ex(x,y,z) = " << boundary_functions.value<0>(x, y, z, 0.0) << std::endl;
                                std::cout << "f_x_computed(x,y,z) = " << vec.value(0, i, j, k) << std::endl;
                                std::cout << "f'_x_ex(x,y,z) = " << exact_first << std::endl;
                                std::cout << "f'_x_computed(x,y,z) = " << numerical_first << std::endl<< std::endl;
                            }
                        }
                        else if (comp == 1)
                        {
                            if(std::abs(exact_first - numerical_first) > 1e-4)
                            {
                                std::cout << "Mismatch detected!" << std::endl;
                                std::cout << std::scientific << std::setprecision(15);
                                std::cout << "x=" << x << ", y=" << y << ", z=" << z << std::endl;
                                std::cout << "i=" << i << ", j=" << j << ", k=" << k << ", comp=" << comp << std::endl;
                                std::cout << "f_y_ex(x,y,z) = " << boundary_functions.value<1>(x, y, z, 0.0) << std::endl;
                                std::cout << "f_y_computed(x,y,z) = " << vec.value(1, i, j, k) << std::endl;
                                std::cout << "f'_y_ex(x,y,z) = " << exact_first << std::endl;
                                std::cout << "f'_y_computed(x,y,z) = " << numerical_first << std::endl<< std::endl;
                            }
                        }
                        else if (comp == 2)
                        {
                            if(std::abs(exact_first - numerical_first) > 1e-4)
                            {
                                std::cout << "Mismatch detected!" << std::endl;
                                std::cout << std::scientific << std::setprecision(15);
                                std::cout << "x=" << x << ", y=" << y << ", z=" << z << std::endl;
                                std::cout << "i=" << i << ", j=" << j << ", k=" << k << ", comp=" << comp << std::endl;
                                std::cout << "f_z_ex(x,y,z) = " << boundary_functions.value<2>(x, y, z, 0.0) << std::endl;
                                std::cout << "f_z_computed(x,y,z) = " << vec.value(2, i, j, k) << std::endl;
                                std::cout << "f'_z_ex(x,y,z) = " << exact_first << std::endl;
                                std::cout << "f'_z_computed(x,y,z) = " << numerical_first << std::endl<< std::endl;
                            }
                            
                        }

                        l2_error_first += (numerical_first - exact_first) * (numerical_first - exact_first);
                        l2_error_second += (numerical_second - exact_second) * (numerical_second - exact_second);
                    }
                }
            }
        }

        l2_error_first = std::sqrt(l2_error_first * Real(dx * dy * dz));
        l2_error_second = std::sqrt(l2_error_second * Real(dx * dy * dz));
        std::cout << Nx << "\t\t" << l2_error_first << "\t\t" << l2_error_second << "\n";
        error_first.emplace_back(l2_error_first);
        error_second.emplace_back(l2_error_second);
    }

    std::cout << "\nConvergence Rates:\n";
    std::cout << "Run\tFirst Deriv Rate\tSecond Deriv Rate\n";
    for (size_t i = 0; i < error_first.size(); ++i)
    {
        if (i == 0)
            std::cout << i << "\t" << "N/A" << "\t\t" << "N/A" << "\n";
        else
        {
            Real rate_first = std::log(error_first[i - 1] / error_first[i]) / std::log(2.0);
            Real rate_second = std::log(error_second[i - 1] / error_second[i]) / std::log(2.0);
            std::cout << i << "\t" << rate_first << "\t\t" << rate_second << "\n";
        }
    }

    return 0;
}