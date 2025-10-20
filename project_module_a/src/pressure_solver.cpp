#include "Navier_stokes_brinkmann.hpp"
#include "ScalarVariables.hpp"

void NavierStokesBrinkmann::pressure_solve()
{
    // First we compute the right-hand side for the pressure equation
    std::vector<Real> rhs_pressure.reserve(Nx * Ny * Nz);
    for (size_t index = 0; index < Nx * Ny * Nz; ++index)
    {
        rhs_pressure.emplace_back((-1 / dt) * velocity_solution.divergence(index));
    }

    solve_linear_systems(rhs_pressure, a, b, c, psi);
    solve_linear_systems(psi, a, b, c, phi);
    solve_linear_systems(phi, a, b, c, other_phi);

    pressure_solution += other_phi;
};