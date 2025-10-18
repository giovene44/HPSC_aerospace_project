#ifndef SCALARVARIABLES_HPP
#define SCALARVARIABLES_HPP

#include "variables.hpp"
#include <vector>
#include <iostream>

class ScalarVariables
{
public:
    ScalarVariables(Dim Nx, Dim Ny, Dim Nz)
        : Nx(Nx), Ny(Ny), Nz(Nz)
    {
        pressure_data.resize(Nx * Ny * Nz, 0.0f);
    };

    Real &set(Dim i, Dim j, Dim k)
    { // This set works like set(i,j,k) = value;
        return pressure_data[i + j * Nx + k * Nx * Ny];
    }

    Real get(Dim i, Dim j, Dim k) const
    {
        return pressure_data[i + j * Nx + k * Nx * Ny];
    }

    Real getGradient_x(Dim i, Dim j, Dim k) const
    {
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

    Real getGradient_z(Dim i, Dim j, Dim k) const
    {
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