#ifndef SCALARVARIABLES_HPP
#define SCALARVARIABLES_HPP

#include "variables.hpp"
#include <vector>
#include <iostream>
#include <stdexcept>

class ScalarVariables
{
public:
    /**
     * Constructor 1:
     * Initializes an empty scalar field of given dimensions
     * and fills it with zeros.
     */
    ScalarVariables(const Dim Nx, const Dim Ny, const Dim Nz)
        : Nx(Nx), Ny(Ny), Nz(Nz)
    {
        size_t total = static_cast<size_t>(Nx) * static_cast<size_t>(Ny) * static_cast<size_t>(Nz);
        data.assign(total, Real(0));
    }

    /**
     * Constructor 2:
     * Initializes the scalar field with an existing data vector.
     * The vector must have size Nx * Ny * Nz.
     */
    ScalarVariables(const Dim Nx, const Dim Ny, const Dim Nz, const std::vector<Real> &input_data)
        : Nx(Nx), Ny(Ny), Nz(Nz)
    {
        if (input_data.size() != static_cast<size_t>(Nx * Ny * Nz))
        {
            throw std::invalid_argument(
                "Error in ScalarVariables constructor: input_data size does not match Nx*Ny*Nz");
        }
        data = input_data;
    }

    // --- Operators ---
    ScalarVariables operator+(const ScalarVariables &other) const
    {
        ScalarVariables result(Nx, Ny, Nz);
        for (Dim index = 0; index < Nx * Ny * Nz; ++index)
        {
            result.set(index) = this->get(index) + other.get(index);
        }
        return result;
    }

    ScalarVariables &operator+=(const ScalarVariables &other)
    {
        for (Dim index = 0; index < Nx * Ny * Nz; ++index)
        {
            this->set(index) += other.get(index);
        }
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
        auto lhs = i % Nx > 0 ? get(i - 1, j, k) : 0.0f;
        auto rhs = i % Nx < Nx - 1 ? get(i + 1, j, k) : 0.0f;
        return (rhs - lhs) * 0.5f;
    }

    Real getGradient_x(Dim index) const
    {
        Dim i = index % Nx;
        Dim j = (index / Nx) % Ny;
        Dim k = index / (Nx * Ny);
        auto lhs = i % Nx > 0 ? get(i - 1, j, k) : 0.0f;
        auto rhs = i % Nx < Nx - 1 ? get(i + 1, j, k) : 0.0f;
        return (rhs - lhs) * 0.5f;
    }

    Real getGradient_y(Dim i, Dim j, Dim k) const
    {
        auto lhs = j % Ny > 0 ? get(i, j - 1, k) : 0.0f;
        auto rhs = j % Ny < Ny - 1 ? get(i, j + 1, k) : 0.0f;
        return (rhs - lhs) * 0.5f;
    }

    Real getGradient_y(Dim index) const
    {
        Dim i = index % Nx;
        Dim j = (index / Nx) % Ny;
        Dim k = index / (Nx * Ny);
        auto lhs = j % Ny > 0 ? get(i, j - 1, k) : 0.0f;
        auto rhs = j % Ny < Ny - 1 ? get(i, j + 1, k) : 0.0f;
        return (rhs - lhs) * 0.5f;
    }

    Real getGradient_z(Dim i, Dim j, Dim k) const
    {
        auto lhs = k % Nz > 0 ? get(i, j, k - 1) : 0.0f;
        auto rhs = k % Nz < Nz - 1 ? get(i, j, k + 1) : 0.0f;
        return (rhs - lhs) * 0.5f;
    }

    Real getGradient_z(Dim index) const
    {
        Dim i = index % Nx;
        Dim j = (index / Nx) % Ny;
        Dim k = index / (Nx * Ny);
        auto lhs = k % Nz > 0 ? get(i, j, k - 1) : 0.0f;
        auto rhs = k % Nz < Nz - 1 ? get(i, j, k + 1) : 0.0f;
        return (rhs - lhs) * 0.5f;
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
};

#endif // SCALARVARIABLES_HPP
