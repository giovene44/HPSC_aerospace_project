#ifndef SCALARVARIABLE_HPP
#define SCALARVARIABLE_HPP

#include "Variables.hpp"
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

    Real get(Dim i, Dim j, Dim k) const
    {
        return data[i + j * Nx + k * Nx * Ny];
    }

    Real get(Dim index) const
    {
        return data[index];
    }

    // --- Gradients ---
    Real getGradient_x(Dim i, Dim j, Dim k) const
    {
        auto lhs = i % Nx > 0 ? get(i, j, k) : 0.0f;
        auto rhs = i % Nx < Nx - 1 ? get(i + 1, j, k) : 0.0f;
        return (rhs - lhs) * 0.5f;
    }

    Real getGradient_x(Dim index) const
    {
        Dim i = index % Nx;
        Dim j = (index / Nx) % Ny;
        Dim k = index / (Nx * Ny);
        return getGradient_x(i, j, k);
    }

    Real getGradient_y(Dim i, Dim j, Dim k) const
    {
        auto lhs = j % Ny > 0 ? get(i, j, k) : 0.0f;
        auto rhs = j % Ny < Ny - 1 ? get(i, j + 1, k) : 0.0f;
        return (rhs - lhs) * 0.5f;
    }

    Real getGradient_y(Dim index) const
    {
        Dim i = index % Nx;
        Dim j = (index / Nx) % Ny;
        Dim k = index / (Nx * Ny);
        return getGradient_y(i, j, k);
    }

    Real getGradient_z(Dim i, Dim j, Dim k) const
    {
        auto lhs = k % Nz > 0 ? get(i, j, k) : 0.0f;
        auto rhs = k % Nz < Nz - 1 ? get(i, j, k + 1) : 0.0f;
        return (rhs - lhs) * 0.5f;
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
        for (Dim i = 0; i < Nx; ++i)
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
        for (Dim i = 0; i < Nx; ++i)
        {
            for (Dim j = 0; j < Ny; ++j)
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
        for (Dim i = 0; i < Nx; ++i)
        {
            for (Dim j = 0; j < Ny; ++j)
            {
                for (Dim k = 0; k < Nz; ++k)
                {
                    gradient.set(i, j, k) = getGradient_z(i, j, k);
                }
            }
        }
        return gradient;
    }

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
