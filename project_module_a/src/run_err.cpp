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

std::pair<Real, Real> single_run(
    Dim Nx_in, Dim Ny_in, Dim Nz_in, Real dt_in,
    Real dx_in, Real dy_in, Real dz_in, Real T_final,
    const std::string &u_boundary_file,
    const std::string &p_boundary_file)
{
    // 1) PARSER RETRIEVAL
    auto &parser = ParseInput::getInstance();

    // Get explicit physics (no Re)
    Real nu = parser.nu;

    // Retrieve functions
    auto forcing_func = parser.get_forcing_function();
    auto k_func = parser.get_k_function();
    auto u_exact_func = parser.get_exact_velocity_function();
    auto p_exact_func = parser.get_exact_pressure_function();

    // 2) MANUFACTURED SOLUTION
    ManufacturedSolution mms(
        u_exact_func, p_exact_func, k_func, forcing_func,
        nu);

    std::cout << "Manufactured solution initialized.\n";

    // 3) SOLVER INITIALIZATION
    NavierStokesBrinkmann nsb_solver(
        Nx_in, Ny_in, Nz_in,
        dt_in, T_final,
        forcing_func,
        k_func,
        u_boundary_file,
        p_boundary_file,
        dx_in, dy_in, dz_in,
        nu);

    std::cout << "Solver initialized (nu=" << nu << ").\n";

    // 4) RUN SOLVER
    nsb_solver.solve(mms);
    std::cout << "\nSolver run completed.\n";

    // 5) COMPUTE ERRORS
    Real err_u = 0.0, err_p = 0.0;
    Real norm_u = 0.0, norm_p = 0.0;
    Real dV = dx_in * dy_in * dz_in;

    std::cout << "Computing Errors at T = " << T_final << "...\n";

    for (Dim k = 0; k < Nz_in; ++k)
    {
        for (Dim j = 0; j < Ny_in; ++j)
        {
            for (Dim i = 0; i < Nx_in; ++i)
            {
                Real x = i * dx_in;
                Real y = j * dy_in;
                Real z = k * dz_in;

                auto u_ex_x = mms.velocity(x+dx_in*0.5, y, z, T_final);
                auto u_ex_y = mms.velocity(x, y+dy_in*0.5, z, T_final);
                auto u_ex_z = mms.velocity(x, y, z+dz_in*0.5, T_final);
                std::vector<Real> u_ex = {u_ex_x[0], u_ex_y[1], u_ex_z[2]};


                Real p_ex = mms.pressure(x, y, z, T_final);

                Real ux = nsb_solver.velocity_solution.value(0, i, j, k);
                Real uy = nsb_solver.velocity_solution.value(1, i, j, k);
                Real uz = nsb_solver.velocity_solution.value(2, i, j, k);
                Real pN = nsb_solver.pressure_solution.get(i, j, k);

                Real dux = ux - u_ex[0];
                Real duy = uy - u_ex[1];
                Real duz = uz - u_ex[2];

                err_u += dux * dux + duy * duy + duz * duz;
                norm_u += u_ex[0] * u_ex[0] + u_ex[1] * u_ex[1] + u_ex[2] * u_ex[2];

                err_p += (pN - p_ex) * (pN - p_ex);
                norm_p += p_ex * p_ex;


            }
        }
    }

    err_u = std::sqrt(err_u * dV);
    norm_u = std::sqrt(norm_u * dV);
    err_p = std::sqrt(err_p * dV);
    norm_p = std::sqrt(norm_p * dV);

    Real rel_err_u = (norm_u > 1e-12) ? err_u / norm_u : err_u;
    Real rel_err_p = (norm_p > 1e-12) ? err_p / norm_p : err_p;

    std::cout << "=============================\n";
    std::cout << "   MMS Accuracy Results      \n";
    std::cout << "=============================\n";
    std::cout << "Velocity L2 absolute  = " << err_u << "\n";
    std::cout << "Pressure L2 absolute  = " << err_p << "\n";
    std::cout << "Velocity L2 relative  = " << rel_err_u << "\n";
    std::cout << "Pressure L2 relative  = " << rel_err_p << "\n";
    std::cout << "=============================\n";

    return std::make_pair(rel_err_u, rel_err_p);
}

int run_multiple()
{
    try
    {
        // ===============================================================
        // 1) PARSER INPUT INITIALIZATION (MOVED HERE)
        // ===============================================================
        ParseInput &parser = ParseInput::getInstance();
        // The path is relative to the execution directory. Using "./Input/Input.in"
        // assumes the Input folder is a direct subfolder of the execution directory.
        parser.parse_input("./Input/Input.in");

        // Get initial values from the parser (used as base for refinement)
        int num_runs = parser.num_runs;

        Dim N_initial_x = parser.Nx;
        Dim N_initial_y = parser.Ny;
        Dim N_initial_z = parser.Nz;
        Real dt_initial = parser.dt;
        Real T_final = parser.T;

        std::vector<Real> N_values;
        std::vector<Real> dt_values;
        std::vector<Real> errors_u;
        std::vector<Real> errors_p;

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

            Real dt_curr = dt_initial;

            T_final = dt_curr *10;

            // dx/dy/dz must be scaled inversely to N_curr (halved when N_curr is doubled)
            Real dx_curr = parser.DimX / (Real)(Nx_curr - 0.5);
            Real dy_curr = parser.DimY / (Real)(Ny_curr - 0.5);
            Real dz_curr = parser.DimZ / (Real)(Nz_curr - 0.5);
            std::cout << "\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n";
            std::cout << "Running simulation with (Nx, Ny, Nz) = ("
                      << Nx_curr << ", " << Ny_curr << ", " << Nz_curr << ") and dt = "
                      << dt_curr << "\n\n";

            // Call single_run with the explicitly computed values
            auto errors = single_run(
                Nx_curr, Ny_curr, Nz_curr, dt_curr,
                dx_curr, dy_curr, dz_curr, T_final,
                parser.u_boundary_file, parser.p_boundary_file);

            // We track the number of grid points (Nx) and the time step (dt) for plotting
            N_values.emplace_back(Nx_curr);
            dt_values.emplace_back(dt_curr);
            errors_u.emplace_back(errors.first);
            errors_p.emplace_back(errors.second);
        }

        // ===============================================================
        // WRITE VELOCITY ERROR FILE (NO N^-2 COLUMN)
        // ===============================================================
        std::ofstream file_u("velocity_error.dat");
        file_u << "# N\tdt\tError_U\n";
        for (size_t i = 0; i < dt_values.size(); i++)
        {
            file_u << N_values[i] << "\t"
                   << dt_values[i] << "\t"
                   << errors_u[i] << "\n";
        }
        file_u.close();

        // ===============================================================
        // WRITE PRESSURE ERROR FILE (NO N^-2 COLUMN)
        // ===============================================================
        std::ofstream file_p("pressure_error.dat");
        file_p << "# N\tdt\tError_P\n";
        for (size_t i = 0; i < dt_values.size(); i++)
        {
            file_p << N_values[i] << "\t"
                   << dt_values[i] << "\t"
                   << errors_p[i] << "\n";
        }
        file_p.close();

        std::cout << "\nData files created: velocity_error.dat, pressure_error.dat\n";
        std::cout << "Skipping automated plotting (python script call removed).\n";

        return 0;
    }
    catch (const std::runtime_error &e)
    {
        std::cerr << "FATAL ERROR in run_multiple: " << e.what() << std::endl;
        return 1;
    }
};
