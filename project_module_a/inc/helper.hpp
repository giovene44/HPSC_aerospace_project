#ifndef HELPER_HPP
#define HELPER_HPP
#include <iomanip> // Required for setw and setprecision
#include "VectorVariable.hpp"

/**
 * @brief Helper to print vector values.
 * Layout: Prints grid slices layer by layer (z-axis).
 * Format: (v_x, v_y, v_z)
 */
inline void print_vector(const VectorVariable &vec, std::ostream &os = std::cout, int precision = 4)
{
    Dim Nx = vec.get_Nx();
    Dim Ny = vec.get_Ny();
    Dim Nz = vec.get_Nz();

    os << std::fixed << std::setprecision(precision);

    for (Dim k = 0; k < Nz; ++k)
    {
        os << "--- Layer z = " << k << " ---\n";
        for (Dim j = 0; j < Ny; ++j)
        {
            os << "Row " << std::setw(2) << j << ": ";
            for (Dim i = 0; i < Nx; ++i)
            {
                Real vx = vec.value(0, i, j, k);
                Real vy = vec.value(1, i, j, k);
                Real vz = vec.value(2, i, j, k);

                os << "("
                   << std::setw(precision + 3) << vx << ", "
                   << std::setw(precision + 3) << vy << ", "
                   << std::setw(precision + 3) << vz
                   << ") ";
            }
            os << "\n";
        }
        os << "\n";
    }
    os.flush();
}

/**
 * @brief Helper to print scalar values.
 * Layout: Prints grid slices layer by layer (z-axis).
 */
inline void print_scalar(const ScalarVariable &scalar, std::ostream &os = std::cout, int precision = 4)
{
    Dim Nx = scalar.get_Nx();
    Dim Ny = scalar.get_Ny();
    Dim Nz = scalar.get_Nz();

    os << std::fixed << std::setprecision(precision);

    for (Dim k = 0; k < Nz; ++k)
    {
        os << "--- Layer z = " << k << " ---\n";
        for (Dim j = 0; j < Ny; ++j)
        {
            os << "Row " << std::setw(2) << j << ": ";
            for (Dim i = 0; i < Nx; ++i)
            {
                os << std::setw(precision + 3) << scalar.get(i, j, k) << " ";
            }
            os << "\n";
        }
        os << "\n";
    }
    os.flush();
}

void print_forcing_function(
    const std::function<std::vector<Real>(Real, Real, Real, Real)> &func,
    Dim Nx, Dim Ny, Dim Nz,
    Real dx, Real dy, Real dz,
    Real t, // Time is required for this function
    std::ostream &os = std::cout,
    int precision = 4)
{
    if (!func)
    {
        os << "forcing_function is empty/undefined.\n";
        return;
    }

    os << std::fixed << std::setprecision(precision);
    os << "--- Sampling forcing_function at t=" << t << " ---\n";

    for (Dim k = 0; k < Nz; ++k)
    {
        Real z = k * dz;
        os << "--- Layer z = " << k << " (phys: " << z << ") ---\n";
        for (Dim j = 0; j < Ny; ++j)
        {
            Real y = j * dy;
            os << "Row " << std::setw(2) << j << ": ";
            for (Dim i = 0; i < Nx; ++i)
            {
                Real x = i * dx;

                // Evaluate function
                std::vector<Real> result = func(x, y, z, t);

                // Safety check for vector size
                Real vx = (result.size() > 0) ? result[0] : 0.0;
                Real vy = (result.size() > 1) ? result[1] : 0.0;
                Real vz = (result.size() > 2) ? result[2] : 0.0;

                os << "("
                   << std::setw(precision + 3) << vx << ", "
                   << std::setw(precision + 3) << vy << ", "
                   << std::setw(precision + 3) << vz
                   << ") ";
            }
            os << "\n";
        }
        os << "\n";
    }
    os.flush();
}

#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>

// Make sure your typedefs (Dim, Real) are available here
// or replace them with int/double

void print_k_function(
    const std::function<Real(Real, Real, Real)> &func,
    Dim Nx, Dim Ny, Dim Nz,
    Real dx, Real dy, Real dz,
    std::ostream &os = std::cout,
    int precision = 4)
{
    if (!func)
    {
        os << "k_function is empty/undefined.\n";
        return;
    }

    os << std::fixed << std::setprecision(precision);
    os << "--- Sampling k_function over grid ---\n";

    for (Dim k = 0; k < Nz; ++k)
    {
        Real z = k * dz;
        os << "--- Layer z = " << k << " (phys: " << z << ") ---\n";
        for (Dim j = 0; j < Ny; ++j)
        {
            Real y = j * dy;
            os << "Row " << std::setw(2) << j << ": ";
            for (Dim i = 0; i < Nx; ++i)
            {
                Real x = i * dx;
                Real val = func(x, y, z); // Evaluate function
                os << std::setw(precision + 3) << val << " ";
            }
            os << "\n";
        }
        os << "\n";
    }
    os.flush();
}

#endif // HELPER_HPP