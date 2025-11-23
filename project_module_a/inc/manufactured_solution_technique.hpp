#pragma once
#include <cmath>
#include <vector>
#include <functional>
#include "Variables.hpp"

using VectorFieldFunction = std::function<std::vector<Real>(Real, Real, Real, Real)>;
using ScalarFieldFunction = std::function<Real(Real, Real, Real, Real)>;
using CoefficientFunction = std::function<Real(Real, Real, Real)>;

class ManufacturedSolution
{
public:
    // Constructor NO LONGER takes Grid dimensions (Nx, dx, etc.)
    ManufacturedSolution(
        VectorFieldFunction u_func,
        ScalarFieldFunction p_func,
        CoefficientFunction k_func,
        VectorFieldFunction f_func,
        Real nu,
        Real x0 = 0.0f, Real y0 = 0.0f, Real z0 = 0.0f)
        : x0(x0), y0(y0), z0(z0),
          nu(nu),
          u_exact(u_func),
          p_exact(p_func),
          k_exact(k_func),
          f_exact(f_func)
    {
    }

    // Wrappers
    std::vector<Real> velocity(Real x, Real y, Real z, Real t) const
    {
        if (u_exact)
            return u_exact(x + x0, y + y0, z + z0, t);
        return {0.0, 0.0, 0.0};
    }

    Real pressure(Real x, Real y, Real z, Real t) const
    {
        if (p_exact)
            return p_exact(x + x0, y + y0, z + z0, t);
        return 0.0;
    }

    Real coefficient(Real x, Real y, Real z) const
    {
        if (k_exact)
            return k_exact(x + x0, y + y0, z + z0);
        return 0.0;
    }

    std::vector<Real> forcing(Real x, Real y, Real z, Real t) const
    {
        if (f_exact)
            return f_exact(x + x0, y + y0, z + z0, t);
        return {0.0, 0.0, 0.0};
    }

    Real get_Nu() const { return nu; }

private:
    // Domain offsets only (if your domain doesn't start at 0,0,0)
    Real x0, y0, z0;
    Real nu;

    VectorFieldFunction u_exact;
    ScalarFieldFunction p_exact;
    CoefficientFunction k_exact;
    VectorFieldFunction f_exact;
};