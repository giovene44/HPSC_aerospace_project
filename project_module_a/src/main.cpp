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
    auto forcing_func = [mms](Real x, Real y, Real z, Real t) -> std::vector<Real>
    {
        return mms.forcing(x, y, z, t);
    };

    auto k_func = [mms](Real x, Real y, Real z) -> Real
    {
        return mms.coefficient(x, y, z);
    };

    // ===============================================================
    // 3) SOLVER INITIALIZATION
    // ===============================================================
    NavierStokesBrinkmann solver(Nx, Ny, Nz, dt, T_final,
                                 forcing_func, k_func,
                                 dx, dy, dz, Re);

    // ===============================================================
    // 5) RUN SOLVER
    // ===============================================================
    solver.solve();

    // ===============================================================
    // 6) COMPUTE L2 ERROR AGAINST MMS SOLUTION AT t = T_final
    // ===============================================================

    Real err_u = 0.0, err_p = 0.0;
    Real norm_u = 0.0, norm_p = 0.0;

    for (Dim k = 0; k < Nz; ++k)
        for (Dim j = 0; j < Ny; ++j)
            for (Dim i = 0; i < Nx; ++i)
            {
                // Exact MMS at final time
                Real uxE = mms.velocity(i * dx, j * dy, k * dz, T_final)[0];
                Real uyE = mms.velocity(i * dx, j * dy, k * dz, T_final)[1];
                Real uzE = mms.velocity(i * dx, j * dy, k * dz, T_final)[2];
                Real pE = mms.pressure(i * dx, j * dy, k * dz);

                // Numerical
                Real ux = solver.velocity_solution.value(0, i, j, k);
                Real uy = solver.velocity_solution.value(1, i, j, k);
                Real uz = solver.velocity_solution.value(2, i, j, k);
                Real pN = solver.pressure_solution.get(i, j, k);

                // Velocity error
                err_u += (ux - uxE) * (ux - uxE) + (uy - uyE) * (uy - uyE) + (uz - uzE) * (uz - uzE);

                norm_u += uxE * uxE + uyE * uyE + uzE * uzE;

                // Pressure error
                err_p += (pN - pE) * (pN - pE);
                norm_p += pE * pE;
            }

    err_u = std::sqrt(err_u);
    norm_u = std::sqrt(norm_u);
    err_p = std::sqrt(err_p);
    norm_p = std::sqrt(norm_p);

    Real rel_err_u = err_u / norm_u;
    Real rel_err_p = err_p / norm_p;

    std::cout << "\n=============================\n";
    std::cout << "   MMS Accuracy Results\n";
    std::cout << "=============================\n";
    std::cout << "Velocity L2 error     = " << err_u << "\n";
    std::cout << "Velocity L2 relative  = " << rel_err_u << "\n";
    std::cout << "Pressure L2 error     = " << err_p << "\n";
    std::cout << "Pressure L2 relative  = " << rel_err_p << "\n";

    return 0;
}
