#include <iostream>
#include <cmath>
#include <cstdlib> // for system()
#include <utility> // for std::pair
#include <fstream>
#include <iomanip>

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
#include <chrono>

std::pair<std::pair<Real, Real>, std::pair<Real, Real>> single_run(
    Dim Nx_in, Dim Ny_in, Dim Nz_in, Real dt_in,
    Real dx_in, Real dy_in, Real dz_in, Real T_final,
    const std::string &u_boundary_file,
    const std::string &p_boundary_file,
    Real &time_out, bool openMP = false)
{
    // 1) PARSER RETRIEVAL
    auto &parser = ParseInput::getInstance();

    // Get explicit physics (no Re)
    Real nu = parser.nu;

    // Retrieve functions
    auto forcing_func = parser.get_forcing_function();
    auto forcing_func_bf = parser.get_forcing_function_bf();
    auto k_func = parser.get_k_function();
    auto u_exact_func = parser.get_exact_velocity_function();
    auto p_exact_func = parser.get_exact_pressure_function();
    auto p_exact_bf = parser.get_exact_pressure_function_bf();

    // 2) MANUFACTURED SOLUTION
    ManufacturedSolution mms(
        u_exact_func, p_exact_func, k_func, forcing_func,
        nu);

    std::cout << "Manufactured solution initialized.\n";

    // 3) SOLVER INITIALIZATION
    NavierStokesBrinkmann nsb_solver(
        Nx_in, Ny_in, Nz_in,
        dt_in, T_final,
        forcing_func_bf,
        k_func,
        u_boundary_file,
        p_boundary_file,
        p_exact_bf,
        dx_in, dy_in, dz_in,
        nu);

    std::cout << "Solver initialized (nu=" << nu << ").\n";

    // 4) RUN SOLVER
    Real time_out_auto = nsb_solver.solve(mms, openMP);
    time_out = time_out_auto;
    std::cout << "\nSolver run completed.\n";

    // 5) COMPUTE ERRORS

    T_final += Real(1.5);

    std::cout << "Computing Errors at T = " << T_final << "...\n";

    auto errors = nsb_solver.compute_L2_errors(
        nsb_solver.u_0,
        nsb_solver.p_0,
        mms,
        T_final);
    Real err_u = errors.first.first;
    Real err_p = errors.second.first;
    Real rel_err_u = errors.first.second;
    Real rel_err_p = errors.second.second;

    std::cout << "=============================\n";
    std::cout << "   MMS Accuracy Results      \n";
    std::cout << "=============================\n";
    std::cout << "Velocity L2 absolute  = " << err_u << "\n";
    std::cout << "Pressure L2 absolute  = " << err_p << "\n";
    std::cout << "Velocity L2 relative  = " << rel_err_u << "\n";
    std::cout << "Pressure L2 relative  = " << rel_err_p << "\n";
    std::cout << "=============================\n";

    return std::make_pair(std::make_pair(err_u, rel_err_u), std::make_pair(err_p, rel_err_p));
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
        std::vector<Real> errors_rel_u;
        std::vector<Real> errors_rel_p;
        std::vector<Real> time_values;
        std::vector<Real> time_speedUps;

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

            Real dt_curr = dt_initial; //* refinement_factor;

            // dx/dy/dz must be scaled inversely to N_curr (halved when N_curr is doubled)
            Real dx_curr = parser.DimX / (Real)(Nx_curr - 0.5);
            Real dy_curr = parser.DimY / (Real)(Ny_curr - 0.5);
            Real dz_curr = parser.DimZ / (Real)(Nz_curr - 0.5);
            std::cout << "\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n";
            std::cout << "Running simulation with (Nx, Ny, Nz) = ("
                      << Nx_curr << ", " << Ny_curr << ", " << Nz_curr << ") and dt = "
                      << dt_curr << "\n\n";

            // Call single_run with the explicitly computed values
            Real time_curr = 0.0;
            bool openMP = false;
            auto errors = single_run(
                Nx_curr, Ny_curr, Nz_curr, dt_curr,
                dx_curr, dy_curr, dz_curr, T_final,
                parser.u_boundary_file, parser.p_boundary_file,
                time_curr, openMP);

            // We track the number of grid points (Nx) and the time step (dt) for plotting
            N_values.emplace_back(Nx_curr);
            dt_values.emplace_back(dt_curr);
            errors_u.emplace_back(errors.first.first);
            errors_p.emplace_back(errors.second.first);
            errors_rel_u.emplace_back(errors.first.second);
            errors_rel_p.emplace_back(errors.second.second);
            time_values.emplace_back(time_curr);

            openMP = true;
            time_curr = 0.0;
            errors = single_run(
                Nx_curr, Ny_curr, Nz_curr, dt_curr,
                dx_curr, dy_curr, dz_curr, T_final,
                parser.u_boundary_file, parser.p_boundary_file,
                time_curr, openMP);
            auto time_speedUp = time_values.back() / time_curr;
            time_speedUps.emplace_back(time_speedUp);
        }

        std::cout << "\n=============================\n";
        std::cout << "   Convergence Analysis      \n";
        std::cout << "=============================\n";

        // Open file for writing
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time), "%Y-%m-%d_%H-%M-%S");
        std::string filename = "OUTPUT/Convergence_Analysis_" + ss.str() + ".dat";
        std::system("mkdir -p OUTPUT");
        std::ofstream convergence_file(filename);

        std::string header = "Nx\t\tdx\tdt\t\tnsteps\t\tL2_u_abs\t\tL2_p_abs\t\tL2_u_rel\t\tL2_p_rel\t\tTime\t\tTime_SpeedUp\t\tRate_u\t\tRate_p\n";
        std::cout << header;
        convergence_file << header;

        for (size_t i = 0; i < N_values.size(); ++i)
        {
            Real dx = parser.DimX / (Real)(N_values[i] - 0.5);
            Real dx_prev = (i > 0) ? parser.DimX / (Real)(N_values[i - 1] - 0.5) : 0.0;
            Dim nsteps = (Dim)(T_final / dt_values[i]);

            Real rate_u = (i > 0) ? std::log(errors_u[i - 1] / errors_u[i]) / std::log(dx_prev / dx) : 0.0;
            Real rate_p = (i > 0) ? std::log(errors_p[i - 1] / errors_p[i]) / std::log(dx_prev / dx) : 0.0;

            std::ostringstream oss;
            oss << std::fixed << std::setprecision(0)
                << N_values[i] << "\t"
                << std::scientific << std::setprecision(6)
                << dx << "\t"
                << dt_values[i] << "\t"
                << nsteps << "\t\t"
                << errors_u[i] << "\t\t"
                << errors_p[i] << "\t\t"
                << errors_rel_u[i] << "\t\t"
                << errors_rel_p[i] << "\t\t"
                << time_values[i] << "\t\t"
                << time_speedUps[i] << "\t\t"
                << std::fixed << std::setprecision(2)
                << rate_u << "\t\t"
                << rate_p << "\n";

            std::cout << oss.str();
            convergence_file << oss.str();
        }

        convergence_file.close();
        std::cout << std::defaultfloat;
        std::cout << "=============================\n";
        std::cout << "Results saved to Convergence_Analysis_today_data.dat\n";

        // std::system("python ./utils/plot.py ./velocity_error.dat");
        // std::system("python ./utils/plot.py ./pressure_error.dat");

        return 0;
    }
    catch (const std::runtime_error &e)
    {
        std::cerr << "FATAL ERROR in run_multiple: " << e.what() << std::endl;
        return 1;
    }
};
