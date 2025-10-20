#ifndef SCALARVARIABLES_HPP
#define SCALARVARIABLES_HPP

#include "variables.hpp"
#include <vector>
#include <iostream>

class ScalarVariables
{
public:
    ScalarVariables(const Dim Nx, const Dim Ny, const Dim Nz)
        : Nx(Nx), Ny(Ny), Nz(Nz)
    {
        pressure_data.resize(Nx * Ny * Nz, 0.0f);
    };

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

    
    Real &set(Dim i, Dim j, Dim k)
    { // This set works like set(i,j,k) = value;
        return pressure_data[i + j * Nx + k * Nx * Ny];
    }

    Real &set(Dim index)
    { // This set works like set(index) = value;
        return pressure_data[index];
    }

    Real get(Dim i, Dim j, Dim k) const
    {
        return pressure_data[i + j * Nx + k * Nx * Ny];
    }

    Real get(Dim index) const
    {
        return pressure_data[index];
    }



    Real getGradient_x(Dim i, Dim j, Dim k) const
    {
        auto lhs = i%Nx > 0 ? get(i - 1, j, k) : 0.0f;
        auto rhs = i%Nx < Nx-1 ? get(i + 1, j, k) : 0.0f;

        return (rhs - lhs) * 0.5f;
    }

    Real getGradient_x(Dim index) const
    {
        Dim i = index % Nx;
        Dim j = (index / Nx) % Ny;
        Dim k = index / (Nx * Ny);

        auto lhs = i%Nx > 0 ? get(i - 1, j, k) : 0.0f;
        auto rhs = i%Nx < Nx-1 ? get(i + 1, j, k) : 0.0f;

        return (rhs - lhs) * 0.5f;
    }

    Real getGradient_y(Dim i, Dim j, Dim k) const
    {
        auto lhs = j%Ny > 0 ? get(i, j - 1, k) : 0.0f;
        auto rhs = j%Ny < Ny-1 ? get(i, j + 1, k) : 0.0f;

        return (rhs - lhs) * 0.5f;
    }


    Real getGradient_y(Dim index) const
    {
        Dim i = index % Nx;
        Dim j = (index / Nx) % Ny;
        Dim k = index / (Nx * Ny);

        auto lhs = j%Ny > 0 ? get(i, j - 1, k) : 0.0f;
        auto rhs = j%Ny < Ny-1 ? get(i, j + 1, k) : 0.0f;

        return (rhs - lhs) * 0.5f;
    }

    Real getGradient_z(Dim i, Dim j, Dim k) const
    {
        auto lhs = k%Nz > 0 ? get(i, j, k - 1) : 0.0f;
        auto rhs = k%Nz < Nz-1 ? get(i, j, k + 1) : 0.0f;

        return (rhs - lhs) * 0.5f;
    }

    Real getGradient_z(Dim index) const
    {
        Dim i = index % Nx;
        Dim j = (index / Nx) % Ny;
        Dim k = index / (Nx * Ny);

        auto lhs = k%Nz > 0 ? get(i, j, k - 1) : 0.0f;
        auto rhs = k%Nz < Nz-1 ? get(i, j, k + 1) : 0.0f;

        return (rhs - lhs) * 0.5f;
    }

private:
    const Dim Nx;
    const Dim Ny;
    const Dim Nz;
    std::vector<Real> pressure_data;
};

#endif // SCALARVARIABLES_HPP