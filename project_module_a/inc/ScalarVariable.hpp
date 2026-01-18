#ifndef SCALARVARIABLE_HPP
#define SCALARVARIABLE_HPP

#include "Variables.hpp"
#include "BoundaryFunctions.hpp"
#include <vector>
#include <iostream>
#include <stdexcept>

class ScalarVariable
{
public:
    /**
     * Constructor:
     * Initializes an empty scalar field of given dimensions
     * and fills it with zeros.
     */
    ScalarVariable(const Dim Nx, const Dim Ny, const Dim Nz, const Real dx_, const Real dy_, const Real dz_)
        : Nx(Nx), Ny(Ny), Nz(Nz), dx(dx_), dy(dy_), dz(dz_)
    {
        data.assign(Nx * Ny * Nz, Real(0));
    }

    ScalarVariable(const ScalarVariable &other)
        : Nx(other.Nx), Ny(other.Ny), Nz(other.Nz),
          data(other.data), dx(other.dx), dy(other.dy), dz(other.dz)
    {
    }

    /**
     * Constructor 2:
     * Initializes the scalar field with an existing data vector.
     * The vector must have size Nx * Ny * Nz.
     */
    ScalarVariable(const Dim Nx, const Dim Ny, const Dim Nz, const std::vector<Real> &input_data)
        : Nx(Nx), Ny(Ny), Nz(Nz)
    {
        if (input_data.size() != static_cast<size_t>(Nx * Ny * Nz))
        {
            throw std::invalid_argument(
                "Error in ScalarVariable constructor: input_data size does not match Nx*Ny*Nz");
        }
        data = input_data;
    }

    // --- Operators ---
    ScalarVariable operator+(const ScalarVariable &other) const
    {
        ScalarVariable result(Nx, Ny, Nz, dx, dy, dz);
        for (Dim index = 0; index < Nx * Ny * Nz; ++index)
        {
            result.set(index) = this->get(index) + other.get(index);
        }
        return result;
    }

    ScalarVariable operator-(const ScalarVariable &other) const
    {
        ScalarVariable result(Nx, Ny, Nz, dx, dy, dz);
        for (Dim index = 0; index < Nx * Ny * Nz; ++index)
        {
            result.set(index) = this->get(index) - other.get(index);
        }
        return result;
    }

    ScalarVariable &operator+=(const ScalarVariable &other)
    {
        for (Dim index = 0; index < Nx * Ny * Nz; ++index)
        {
            this->set(index) += other.get(index);
        }
        return *this;
    }

    ScalarVariable &operator-=(const ScalarVariable &other)
    {
        for (Dim index = 0; index < Nx * Ny * Nz; ++index)
        {
            this->set(index) -= other.get(index);
        }
        return *this;
    }

    ScalarVariable &operator=(const ScalarVariable &other)
    {
        if (this == &other)
            return *this;

        if (Nx != other.Nx || Ny != other.Ny || Nz != other.Nz)
            throw std::runtime_error("ScalarVariable::operator=: dimension mismatch");

        data = other.data;
        dx = other.dx;
        dy = other.dy;
        dz = other.dz;
        return *this;
    }
    // --- Accessors ---
    Real &set(Dim i, Dim j, Dim k)
    {
        return data[i + j * Nx + k * Nx * Ny];
    }

    Real &set(Dim index)
    {
        return data[index];
    }

    /**
     * @brief Sets all elements in the scalar field to the specified value.
     * @param value The Real value to assign to all elements.
     */
    void set_all(Real value)
    {
        std::fill(data.begin(), data.end(), value);
    }

    void set_all(BoundaryFunctions &other, Real t)
    {
        for (Dim idx = 0; idx < Nx * Ny * Nz; ++idx)
        {
            Dim i = idx % Nx;
            Dim j = (idx / Nx) % Ny;
            Dim k = idx / (Nx * Ny);

            // Convert grid indices to physical coordinates
            Real x = i * dx;
            Real y = j * dy;
            Real z = k * dz;

            this->set(idx) = other.value<0>(x, y, z, t);
        }
    }

    Real get(Dim i, Dim j, Dim k) const
    {
        return data[i + j * Nx + k * Nx * Ny];
    }

    Real get(Dim index) const
    {
        return data[index];
    }

    // --- Gradients ---
    // --- Gradients (Centered Finite Difference - 2nd Order Interior) ---

    Real getGradient_x(Dim i, Dim j, Dim k) const
    {
        // Interior points (Second Order Centered)
        if (i < Nx - 1)
        {
            Real lhs = get(i, j, k);
            Real rhs = get(i + 1, j, k);
            return (rhs - lhs) / dx;
        }
        else 
            return 0.0; // g not used at the boundary!
    }

    Real getGradient_y(Dim i, Dim j, Dim k) const
    {
        // Interior points (Second Order Centered)
        if (j < Ny - 1)
        {
            Real lhs = get(i, j, k);
            Real rhs = get(i, j + 1, k);
            return (rhs - lhs) / dy;
        }
        // Bottom Boundary (Forward Difference - 1st Order)
        else
        {
            return 0.0; // g not used at the boundary!
        }
     
    }

    Real getGradient_z(Dim i, Dim j, Dim k) const
    {
        // Interior points (Second Order Centered)
        if (k < Nz - 1)
        {
            Real lhs = get(i, j, k);
            Real rhs = get(i, j, k + 1);
            return (rhs - lhs) / dz;
        }
        else
        {
            return 0.0; // g not used at the boundary!  
        }
    }

    Real getGradient_x(Dim index) const
    {
        Dim i = index % Nx;
        Dim j = (index / Nx) % Ny;
        Dim k = index / (Nx * Ny);
        return getGradient_x(i, j, k);
    }

    Real getGradient_y(Dim index) const
    {
        Dim i = index % Nx;
        Dim j = (index / Nx) % Ny;
        Dim k = index / (Nx * Ny);
        return getGradient_y(i, j, k);
    }

    Real getGradient_z(Dim index) const
    {
        Dim i = index % Nx;
        Dim j = (index / Nx) % Ny;
        Dim k = index / (Nx * Ny);
        return getGradient_z(i, j, k);
    }

    ScalarVariable getGradient_x() const
    {
        // Overloaded function to compute gradient for all elements
        ScalarVariable gradient(Nx, Ny, Nz, dx, dy, dz);

        gradient.set_all(0.0f); // Initialize to zero

        for (Dim i = 1; i < Nx-1; ++i)
        {
            for (Dim j = 0; j < Ny; ++j)
            {
                for (Dim k = 0; k < Nz; ++k)
                {
                    gradient.set(i, j, k) = getGradient_x(i, j, k);
                }
            }
        }
        return gradient;
    }

    ScalarVariable getGradient_y() const
    {
        // Overloaded function to compute gradient for all elements
        ScalarVariable gradient(Nx, Ny, Nz, dx, dy, dz);
        gradient.set_all(0.0f); // Initialize to zero
        for (Dim i = 0; i < Nx; ++i)
        {
            for (Dim j = 1; j < Ny - 1; ++j)
            {
                for (Dim k = 0; k < Nz; ++k)
                {
                    gradient.set(i, j, k) = getGradient_y(i, j, k);
                }
            }
        }
        return gradient;
    }

    ScalarVariable getGradient_z() const
    {
        // Overloaded function to compute gradient for all elements
        ScalarVariable gradient(Nx, Ny, Nz, dx, dy, dz);
        gradient.set_all(0.0f); // Initialize to zero
        for (Dim i = 0; i < Nx; ++i)
        {
            for (Dim j = 0; j < Ny; ++j)
            {
                for (Dim k = 1; k < Nz - 1; ++k)
                {
                    gradient.set(i, j, k) = getGradient_z(i, j, k);
                }
            }
        }
        return gradient;
    }

    Real second_derivative(int direction, Dim i, Dim j, Dim k) const
    {
        if (direction == 0) // x-direction
        {
            if (i > 0 && i < Nx - 1)
            {
                Real lhs = get(i - 1, j, k);
                Real center = get(i, j, k);
                Real rhs = get(i + 1, j, k);
                return (lhs - 2 * center + rhs) / (dx * dx);
            }
            else
            {
                return 0.0;
            }
        }
        else if (direction == 1) // y-direction
        {
            if (j > 0 && j < Ny - 1)
            {
                Real lhs = get(i, j - 1, k);
                Real center = get(i, j, k);
                Real rhs = get(i, j + 1, k);
                return (lhs - 2 * center + rhs) / (dy * dy);
            }
            else
            {
                return 0.0;
            }
        }
        else if (direction == 2) // z-direction
        {
            if (k > 0 && k < Nz - 1)
            {
                Real lhs = get(i, j, k - 1);
                Real center = get(i, j, k);
                Real rhs = get(i, j, k + 1);
                return (lhs - 2 * center + rhs) / (dz * dz);
            }
            else
            {
                return 0.0;
            }
        }
        else
        {
            throw std::invalid_argument("Invalid direction for second_derivative");
        }
    }

    inline Dim get_Nx() const { return Nx; }
    inline Dim get_Ny() const { return Ny; }
    inline Dim get_Nz() const { return Nz; }
    // --- Utility ---
    size_t size() const { return data.size(); }
    const std::vector<Real> &getData() const { return data; }
    std::vector<Real> &getData() { return data; }

private:
    const Dim Nx;
    const Dim Ny;
    const Dim Nz;
    std::vector<Real> data;
    Real dx, dy, dz;
};

#endif // SCALARVARIABLE_HPP
