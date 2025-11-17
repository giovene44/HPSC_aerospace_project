#include <iostream>
#include <cmath>

#include "manufactured_solution_technique.hpp"
#include "navier_stokes_brinkman.hpp"
#include "ScalarVariable.hpp"
#include "VectorVariable.hpp"

int main()
{
    // ===============================================================
    // 1) GRID PARAMETERS
    // ===============================================================
    Dim Nx = 32, Ny = 32, Nz = 32;
    Real dx = 2.0 * M_PI / Nx;
    Real dy = 2.0 * M_PI / Ny;
    Real dz = 2.0 * M_PI / Nz;

    Real dt = 0.01;
    Real T_final = 0.1;
    Real Re = 100.0;

    // ===============================================================
    // 2) MANUFACTURED SOLUTION INITIALIZATION
    // ===============================================================
    ManufacturedSolution mms(Nx, Ny, Nz, dx, dy, dz, Re);

    Real t0 = 0.0;

    // Initial conditions
    auto forcing_func = [mms](Real x, Real y, Real z, Real t) -> std::vector<Real> {
        return mms.forcing(x, y, z, t);
    };
    
    // ===============================================================
    // 3) SOLVER INITIALIZATION
    // ===============================================================
    NavierStokesBrinkmann solver(Nx, Ny, Nz, dt, T_final,
        forcing_func,
         dx, dy, dz);

    // viscosity field ν = 1/Re
    for (Dim k = 0; k < Nz; ++k)
        for (Dim j = 0; j < Ny; ++j)
            for (Dim i = 0; i < Nx; ++i)
            {
                solver.nu.set(i, j, k) = 1.0 / Re;
                solver.k_field.set(i, j, k) = mms.coefficient(i, j, k); // Permeability field k = 1 (no porous medium)
            }
    solver.initialize_gamma_field();

    // ===============================================================
    // 5) RUN SOLVER
    // ===============================================================
    solver.solve();
    return 0;
}
   