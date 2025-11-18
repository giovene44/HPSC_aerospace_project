#pragma once
#include <cmath>
#include <vector>
#include "Variables.hpp"
#include "VectorVariable.hpp"
#include "ScalarVariable.hpp"

class ManufacturedSolution
{
public:
    ManufacturedSolution(
        Dim Nx_, Dim Ny_, Dim Nz_,
        Real dx_, Real dy_, Real dz_,
        Real reynolds_number = 100.0f,
        Real x0 = 0.0f, Real y0 = 0.0f, Real z0 = 0.0f)
        : Nx(Nx_), Ny(Ny_), Nz(Nz_),
          dx(dx_), dy(dy_), dz(dz_),
          x0(x0), y0(y0), z0(z0),
          Re(reynolds_number)
    {
    }

    // =====================================================================
    //  Pointwise MMS functions
    // =====================================================================
    std::vector<Real> velocity(Real x, Real y, Real z, Real t) const
    {
        std::vector<Real> u(3);

        Real sin_x = std::sin(x);
        Real cos_x = std::cos(x);
        Real sin_y = std::sin(y);
        Real cos_y = std::cos(y);
        Real sin_z = std::sin(z);
        Real cos_z = std::cos(z);
        Real sin_t = std::sin(t);

        u[0] = sin_t * sin_x * sin_y * sin_z;
        u[1] = sin_t * cos_x * cos_y * cos_z;
        u[2] = sin_t * cos_x * sin_y * (sin_z + cos_z);

        return u;
    }

    Real pressure(Real x, Real y, Real z) const
    {
        Real cos_x = std::cos(x);
        Real sin_y = std::sin(y);
        Real sin_z = std::sin(z);
        Real cos_z = std::cos(z);

        return (-3.0f / Re) * cos_x * sin_y * (sin_z - cos_z);
    }

    Real coefficient(Real x, Real y, Real z) const
    {
        return std::sin(x) * std::sin(y) * std::sin(z);
    }

    std::vector<Real> forcing(Real x, Real y, Real z, Real t) const
    {
        std::vector<Real> f(3);

        Real sin_x = std::sin(x);
        Real cos_x = std::cos(x);
        Real sin_y = std::sin(y);
        Real cos_y = std::cos(y);
        Real sin_z = std::sin(z);
        Real cos_z = std::cos(z);
        Real sin_t = std::sin(t);
        Real cos_t = std::cos(t);

        Real k = coefficient(x, y, z);

        Real u = sin_t * sin_x * sin_y * sin_z;
        Real v = sin_t * cos_x * cos_y * cos_z;
        Real w = sin_t * cos_x * sin_y * (sin_z + cos_z);

        f[0] = cos_t * u + (3.0f / Re) * u + k * u +
               (3.0f / Re) * sin_x * sin_y * (sin_z - cos_z);

        f[1] = cos_t * v + (3.0f / Re) * v + k * v -
               (3.0f / Re) * cos_x * cos_y * (sin_z - cos_z);

        f[2] = cos_t * w + (3.0f / Re) * w + k * w -
               (3.0f / Re) * cos_x * sin_y * (cos_z + sin_z);

        return f;
    }

private:
    // Grid parameters
    Dim Nx, Ny, Nz;
    Real dx, dy, dz;

    // Domain offsets
    Real x0, y0, z0;

    // Physical parameter
    Real Re;
};
