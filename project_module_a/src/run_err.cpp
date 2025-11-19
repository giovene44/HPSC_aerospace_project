#include <iostream>
#include <cmath>
#include <cstdlib> // for system()
#include <utility> // for std::pair
#include <fstream>

#include "manufactured_solution_technique.hpp"
#include "navier_stokes_brinkman.hpp"
#include "ScalarVariable.hpp"
#include "VectorVariable.hpp"
#include "ParseInput.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include <functional>

/**
 * @brief Runs a single simulation with specified parameters and computes the L2 errors.
 * * @param Nx_in Grid points in x.
 * @param Ny_in Grid points in y.
 * @param Nz_in Grid points in z.
 * @param dt_in Time step size.
 * @param dx_in Grid spacing in x.
 * @param dy_in Grid spacing in y.
 * @param dz_in Grid spacing in z.
 * @param T_final Final simulation time.
 * @param u_file Initial condition file for velocity.
 * @param p_file Initial condition file for pressure.
 * @return std::pair<Real, Real> Relative L2 errors for velocity (first) and pressure (second).
 */
std::pair<Real, Real> single_run(
    Dim Nx_in, Dim Ny_in, Dim Nz_in, Real dt_in,
    Real dx_in, Real dy_in, Real dz_in, Real T_final,
    const std::string &u_file, const std::string &p_file)
{
    // ===============================================================
    // 1) GRID PARAMETERS
    // ===============================================================
    Real Re = 100.0;

    // ===============================================================
    // 2) MANUFACTURED SOLUTION INITIALIZATION
    // The MMS must be initialized with the current grid parameters
    // ===============================================================
    ManufacturedSolution mms(Nx_in, Ny_in, Nz_in, dx_in, dy_in, dz_in, Re);

    // Initial conditions (forcing and coefficient functions)
    auto forcing_func = [mms](Real x, Real y, Real z, Real t) -> std::vector<Real>
    {
        return mms.forcing(x, y, z, t);
    };

    auto k_func = [mms](Real x, Real y, Real z) -> Real
    {
        return mms.coefficient(x, y, z);
    };

    std::cout << "Manufactured solution initialized.\n";

    // ===============================================================
    // 3) SOLVER INITIALIZATION
    // ===============================================================
    NavierStokesBrinkmann solver(
        Nx_in, Ny_in, Nz_in,
        dt_in, T_final,
        forcing_func,
        k_func,
        u_file,
        p_file,
        dx_in, dy_in, dz_in,
        Re);
    std::cout << "Solver initialized.\n";

    // ===============================================================
    // 5) RUN SOLVER
    // ===============================================================
    solver.solve();
    std::cout << "Solver run completed.\n";

    // ===============================================================
    // 6) COMPUTE L2 ERROR AGAINST MMS SOLUTION AT t = T_final
    // ===============================================================

    Real err_u = 0.0, err_p = 0.0;
    Real norm_u = 0.0, norm_p = 0.0;

    for (Dim k = 0; k < Nz_in; ++k)
        for (Dim j = 0; j < Ny_in; ++j)
            for (Dim i = 0; i < Nx_in; ++i)
            {
                // Exact MMS at final time (uses current dx/dy/dz and T_final)
                Real x_coord = i * dx_in;
                Real y_coord = j * dy_in;
                Real z_coord = k * dz_in;

                Real uxE = mms.velocity(x_coord, y_coord, z_coord, T_final)[0];
                Real uyE = mms.velocity(x_coord, y_coord, z_coord, T_final)[1];
                Real uzE = mms.velocity(x_coord, y_coord, z_coord, T_final)[2];
                Real pE = mms.pressure(x_coord, y_coord, z_coord);

                // Numerical (uses current indices)
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

    return std::make_pair(rel_err_u, rel_err_p);
}
int run_multiple(int num_runs)
{
    try
    {
        // ===============================================================
        // 1) PARSER INPUT INITIALIZATION (MOVED HERE)
        // ===============================================================
        ParseInput &parser = ParseInput::getInstance();
        // The path is relative to the execution directory. Using "./Input/Input.txt"
        // assumes the Input folder is a direct subfolder of the execution directory.
        parser.parse_input("./Input/Input.txt");

        // Get initial values from the parser (used as base for refinement)
        Dim N_initial_x = parser.Nx;
        Dim N_initial_y = parser.Ny;
        Dim N_initial_z = parser.Nz;
        Real dt_initial = parser.dt;
        Real T_final = parser.T;

        // Store dx_initial values from the file, used to derive dx_curr
        Real dx_initial = parser.dx;
        Real dy_initial = parser.dy;
        Real dz_initial = parser.dz;

        std::vector<Real> N_values;
        std::vector<Real> dt_values;
        std::vector<Real> errors_0;
        std::vector<Real> errors_1;

        for (int i = 0; i < num_runs; i++)
        {
            // Calculate refinement factor (2^i)
            Real refinement_factor = std::pow(2, i);

            // =================================================================
            // GRID AND TIME CALCULATION: Starting point is correctly i=0 (factor 1)
            // =================================================================
            Dim Nx_curr = N_initial_x * refinement_factor;
            Dim Ny_curr = N_initial_y * refinement_factor;
            Dim Nz_curr = N_initial_z * refinement_factor;

            Real dt_curr = dt_initial / refinement_factor;

            // dx/dy/dz must be scaled inversely to N_curr (halved when N_curr is doubled)
            Real dx_curr = dx_initial / refinement_factor;
            Real dy_curr = dy_initial / refinement_factor;
            Real dz_curr = dz_initial / refinement_factor;

            std::cout << "\nRunning simulation with (Nx, Ny, Nz) = ("
                      << Nx_curr << ", " << Ny_curr << ", " << Nz_curr << ") and dt = "
                      << dt_curr << "\n\n";

            // Call single_run with the explicitly computed values
            auto errors = single_run(
                Nx_curr, Ny_curr, Nz_curr, dt_curr,
                dx_curr, dy_curr, dz_curr, T_final,
                parser.u_boundary_file, parser.p_boundary_file);

            // We track the number of grid points (Nx) and the time step (dt) for plotting
            N_values.push_back(Nx_curr);
            dt_values.push_back(dt_curr);
            errors_0.push_back(errors.first);
            errors_1.push_back(errors.second);
        }

        // Write error vs N (Spatial Convergence)
        std::ofstream file_N("error_vs_N.dat");
        file_N << "# N\tUErr\n";
        for (size_t i = 0; i < N_values.size(); i++)
        {
            file_N << N_values[i] << "\t" << errors_0[i] << "\t" << "\n";
        }
        file_N.close();

        // Write error vs dt (Temporal Convergence)
        std::ofstream file_dt("error_vs_dt.dat");
        file_dt << "# dt\tPerr\n";
        for (size_t i = 0; i < dt_values.size(); i++)
        {
            file_dt << dt_values[i] << "\t" << errors_1[i] << "\n";
        }
        file_dt.close();

        std::cout << "\nData files created. Generating plots...\n";

        // Call Python script to generate plots
        int result = system("python3 utils/plot.py error_vs_N.dat error_vs_N.png");
        if (result != 0)
        {
            std::cerr << "Warning: Failed to plot error_vs_N.dat\n";
        }

        result = system("python3 utils/plot.py error_vs_dt.dat error_vs_dt.png");
        if (result != 0)
        {
            std::cerr << "Warning: Failed to plot error_vs_dt.dat\n";
        }

        std::cout << "Plots generated successfully!\n";

        return 0;
    }
    catch (const std::runtime_error &e)
    {
        std::cerr << "FATAL ERROR in run_multiple: " << e.what() << std::endl;
        return 1; // Return non-zero to indicate application failure
    }
}