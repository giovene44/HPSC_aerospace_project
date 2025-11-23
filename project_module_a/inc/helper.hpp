#ifndef HELPER_HPP
#define HELPER_HPP
#include <iomanip> // Required for setw and setprecision
#include "VectorVariable.hpp"

/**
 * @brief Helper to print vector values.
 * Layout: Prints grid slices layer by layer (z-axis).
 * Format: (v_x, v_y, v_z)
 */
inline void print_vector(const VectorVariable &vec, std::ostream &os = std::cout, int precision = 8)
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
inline void print_scalar(const ScalarVariable &scalar, std::ostream &os = std::cout, int precision = 8)
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

inline bool are_vectors_equal(const VectorVariable &v1, const VectorVariable &v2,
                              Real tolerance = 1e-6, bool verbose = true)
{
    // 1. Check Dimensions
    if (v1.get_Nx() != v2.get_Nx() ||
        v1.get_Ny() != v2.get_Ny() ||
        v1.get_Nz() != v2.get_Nz())
    {
        if (verbose)
        {
            std::cerr << " [!] Dimension mismatch:\n"
                      << "     Vec1: (" << v1.get_Nx() << ", " << v1.get_Ny() << ", " << v1.get_Nz() << ")\n"
                      << "     Vec2: (" << v2.get_Nx() << ", " << v2.get_Ny() << ", " << v2.get_Nz() << ")\n";
        }
        return false;
    }

    Dim Nx = v1.get_Nx();
    Dim Ny = v1.get_Ny();
    Dim Nz = v1.get_Nz();

    // 2. Iterate through the grid
    for (Dim k = 0; k < Nz; ++k)
    {
        for (Dim j = 0; j < Ny; ++j)
        {
            for (Dim i = 0; i < Nx; ++i)
            {
                // Check X, Y, Z components
                for (int comp = 0; comp < 3; ++comp)
                {
                    Real val1 = v1.value(comp, i, j, k);
                    Real val2 = v2.value(comp, i, j, k);
                    Real diff = std::abs(val1 - val2);

                    if (diff > tolerance)
                    {
                        if (verbose)
                        {
                            char comp_label = (comp == 0) ? 'x' : (comp == 1) ? 'y'
                                                                              : 'z';
                            std::cerr << std::fixed << std::setprecision(6);
                            std::cerr << " [!] Mismatch found at Index (" << i << ", " << j << ", " << k << "):\n"
                                      << "     Component: " << comp_label << "\n"
                                      << "     v1 value : " << val1 << "\n"
                                      << "     v2 value : " << val2 << "\n"
                                      << "     Diff     : " << diff << " (Tolerance: " << tolerance << ")\n";
                        }
                        return false;
                    }
                }
            }
        }
    }

    if (verbose)
        std::cout << " [OK] Vectors are equal.\n";
    return true;
}

// Small helper to keep the main function clean
inline void report_mismatch(int comp, int i, int j, int k, Real grid, Real exact, Real tol)
{
    char label = (comp == 0) ? 'x' : (comp == 1) ? 'y'
                                                 : 'z';
    std::cerr << " [!] Mismatch at Index (" << i << ", " << j << ", " << k << ") Comp " << label << "\n"
              << "     Grid Val : " << grid << "\n"
              << "     Exact Val: " << exact << "\n"
              << "     Diff     : " << std::abs(grid - exact) << " > " << tol << "\n";
}

inline bool are_vector_function_equal(
    const VectorVariable &vec,
    const std::function<std::vector<Real>(Real, Real, Real, Real)> &func,
    Real dx, Real dy, Real dz,
    Real t,
    Real tolerance = 1e-6,
    bool verbose = true)
{
    if (!func)
        return false;

    Dim Nx = vec.get_Nx();
    Dim Ny = vec.get_Ny();
    Dim Nz = vec.get_Nz();

    for (Dim k = 0; k < Nz; ++k)
    {
        for (Dim j = 0; j < Ny; ++j)
        {
            for (Dim i = 0; i < Nx; ++i)
            {
                // Base coordinates
                Real base_x = i * dx;
                Real base_y = j * dy;
                Real base_z = k * dz;

                // --- Check Component 0 (X-Velocity / u) ---
                // Location: (x + dx/2, y, z)
                std::vector<Real> exact_u_vec = func(base_x + dx / 2.0f, base_y, base_z, t);
                Real exact_val_0 = (exact_u_vec.size() > 0) ? exact_u_vec[0] : 0.0;
                Real grid_val_0 = vec.value(0, i, j, k);

                if (std::abs(grid_val_0 - exact_val_0) > tolerance)
                {
                    if (verbose)
                        report_mismatch(0, i, j, k, grid_val_0, exact_val_0, tolerance);
                    return false;
                }

                // --- Check Component 1 (Y-Velocity / v) ---
                // Location: (x, y + dy/2, z)
                std::vector<Real> exact_v_vec = func(base_x, base_y + dy / 2.0f, base_z, t);
                Real exact_val_1 = (exact_v_vec.size() > 1) ? exact_v_vec[1] : 0.0;
                Real grid_val_1 = vec.value(1, i, j, k);

                if (std::abs(grid_val_1 - exact_val_1) > tolerance)
                {
                    if (verbose)
                        report_mismatch(1, i, j, k, grid_val_1, exact_val_1, tolerance);
                    return false;
                }

                // --- Check Component 2 (Z-Velocity / w) ---
                // Location: (x, y, z + dz/2)
                std::vector<Real> exact_w_vec = func(base_x, base_y, base_z + dz / 2.0f, t);
                Real exact_val_2 = (exact_w_vec.size() > 2) ? exact_w_vec[2] : 0.0;
                Real grid_val_2 = vec.value(2, i, j, k);

                if (std::abs(grid_val_2 - exact_val_2) > tolerance)
                {
                    if (verbose)
                        report_mismatch(2, i, j, k, grid_val_2, exact_val_2, tolerance);
                    return false;
                }
            }
        }
    }

    if (verbose)
        std::cout << " [OK] Vector grid matches analytical function (Staggered Grid checked).\n";
    return true;
}

#endif // HELPER_HPP