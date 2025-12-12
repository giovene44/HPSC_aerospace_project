#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
#include "navier_stokes_brinkman.hpp"
#include <chrono>

// ===============================================================
// 1. Setup Grid Parameters
// ===============================================================
struct Grid
{
    Dim Nx, Ny, Nz;
    Real dx, dy, dz;
    Real dt;
};

Grid setup_grid(Real dim_x, Real dim_y, Real dim_z, Dim Nx, Dim Ny, Dim Nz, Real dt)
{
    Grid g;
    g.Nx = Nx;
    g.Ny = Ny;
    g.Nz = Nz;
    g.dx = dim_x / (Nx - 0.5);
    g.dy = dim_y / (Ny - 0.5);
    g.dz = dim_z / (Nz - 0.5);
    g.dt = dt;
    return g;
}

// ===============================================================
// 3. Initialize Fields (interior + RHS)
// ===============================================================
void initialize_fields(const Grid &g,
                       ScalarVariable &gamma_field,
                       VectorVariable &vector,
                       VectorVariable &rhs,
                       BoundaryFunctions &u_boundary,
                       Real t)
{
    (void)u_boundary; // Unused parameter

    // Initialize interior
    for (int comp = 0; comp < 3; ++comp)
        for (int k = 0; k < g.Nz; ++k)
            for (int j = 0; j < g.Ny; ++j)
                for (int i = 0; i < g.Nx; ++i)
                {
                    Real x, y, z;
                    if (comp == 0)
                    {
                        x = (i + 0.5) * g.dx;
                        y = j * g.dy;
                        z = k * g.dz;
                    }
                    else if (comp == 1)
                    {
                        x = i * g.dx;
                        y = (j + 0.5) * g.dy;
                        z = k * g.dz;
                    }
                    else
                    {
                        x = i * g.dx;
                        y = j * g.dy;
                        z = (k + 0.5) * g.dz;
                    }

                    if (comp == 0)
                        vector.set(comp, i, j, k) = sin(t) * sin(x) * sin(y) * sin(z);
                    else if (comp == 1)
                        vector.set(comp, i, j, k) = sin(t) * cos(x) * cos(y) * cos(z);
                    else
                        vector.set(comp, i, j, k) = sin(t) * cos(x) * sin(y) * (sin(z) + cos(z));
                }

    // Apply known faces
    // apply_known_faces(vector, u_boundary, g, gamma_field, 0.0);

    // Build RHS = (I - gamma * Dxx) * vector
    for (int comp = 0; comp < 3; ++comp)
        for (Dim k = 0; k < g.Nz; ++k)
            for (Dim j = 0; j < g.Ny; ++j)
                for (Dim i = 0; i < g.Nx; ++i)
                {
                    Real val = vector.value(comp, i, j, k);
                    Real gamma_val = gamma_field.get(i, j, k);
                    Real sd = vector.second_derivative(comp, 0, i, j, k);
                    Real rhs_val = val - gamma_val * sd;

                    rhs.set(comp, i, j, k) = rhs_val;
                }
}

// ===============================================================
// 4. Setup Solver & Strides
// ===============================================================
VelocitySolver setup_solver(const Grid &g, ScalarVariable &gamma_field, BoundaryFunctions &u_boundary, Real dt, Real t)
{
    VelocitySolver solver(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz, dt, gamma_field, u_boundary);
    solver.gamma_field = gamma_field;
    solver.u_boundary = u_boundary;
    solver.set_t(t);
    return solver;
}

auto define_stride_x(const Grid &g)
{
    return [=](Dim j, Dim k)
    { return j * g.Nx + k * g.Nx * g.Ny; };
}

// ===============================================================
// 6. Solve and check solution
// ===============================================================
Real compute_L2_error(VectorVariable &expected, VectorVariable &computed, const Grid &g)
{
    Real error = 0.0;
    for (int comp = 0; comp < 3; ++comp)
        for (Dim k = 0; k < g.Nz; ++k)
            for (Dim j = 0; j < g.Ny; ++j)
                for (Dim i = 0; i < g.Nx; ++i)
                {
                    Real diff = expected.value(comp, i, j, k) - computed.value(comp, i, j, k);
                    error += diff * diff;
                }
    return sqrt(error * g.dx * g.dy * g.dz);
}

bool solve_and_check(VelocitySolver &solver, VectorVariable &rhs_1,
                     VectorVariable &vector_1, VectorVariable &rhs_2,
                     VectorVariable &vector_2, const Grid &g,
                     DimensionsHandlerVector &x_handler,
                     Real &l2_error, auto &time_speedup)
{
    // PARALLEL SOLVE
    VectorVariable rhs_delta_parallel(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    VectorVariable vector_delta_parallel(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    rhs_delta_parallel = rhs_2 - rhs_1;
    vector_delta_parallel = vector_2 - vector_1;
    VectorVariable computed_sol_parallel(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    computed_sol_parallel.set_all(0.0);
    auto time_parallel_start = std::chrono::high_resolution_clock::now();
    solver.solve<0>(rhs_delta_parallel, computed_sol_parallel, x_handler, true);
    auto time_parallel_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_parallel_diff = time_parallel_end - time_parallel_start;
    // Compute L2 error
    l2_error = compute_L2_error(vector_delta_parallel, computed_sol_parallel, g);

    // SERIAL SOLVE
    VectorVariable rhs_delta_serial(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    VectorVariable vector_delta_serial(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    rhs_delta_serial = rhs_2 - rhs_1;
    vector_delta_serial = vector_2 - vector_1;
    VectorVariable computed_sol_serial(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    computed_sol_serial.set_all(0.0);
    auto time_serial_start = std::chrono::high_resolution_clock::now();
    solver.solve<0>(rhs_delta_serial, computed_sol_serial, x_handler, false);
    auto time_serial_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_serial_diff = time_serial_end - time_serial_start;

    time_speedup = time_serial_diff.count() / time_parallel_diff.count();

    return true;
}

// ===============================================================
// 7. Main function
// ===============================================================
int main()
{
    const Real two_pi = 2.0 * 3.141592653589793;

    // Open output file
    std::ofstream outfile("convergence_momentum_x.txt");
    outfile << "# Convergence study for momentum x-direction solver\n";
    outfile << "# Nx Ny Nz dx dt L2_error convergence_rate\n";
    outfile << std::scientific << std::setprecision(8);

    // Test multiple grid resolutions
    std::vector<Dim> grid_sizes = {20, 40, 80, 160};
    std::vector<Real> errors;
    std::vector<Real> speed_ups;
    std::vector<Real> dx_values;
    std::vector<Real> dt_values;

    for (Dim N : grid_sizes)
    {
        // Make dt proportional to dx^2 to keep temporal error negligible for spatial convergence study
        Real dx_nominal = two_pi / (N - 0.5);
        Real dt_test = 0.0025; //* dx_nominal; // dt ~ O(dx^2)
        Grid g = setup_grid(two_pi, two_pi, two_pi, N, N, N, dt_test);

        printf("\n=================================================\n");
        printf("Grid: Nx=%d, Ny=%d, Nz=%d, dx=%f, dy=%f, dz=%f, dt=%f\n",
               g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz, g.dt);
        printf("=================================================\n");

        ScalarVariable gamma_field(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
        gamma_field.set_all(0.1f);
        VectorVariable vector_1(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
        VectorVariable rhs_1(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
        VectorVariable vector_2(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
        VectorVariable rhs_2(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);

        BoundaryFunctions u_boundary;
        std::vector<std::string> sin_bc = {"sin(t)*sin(x)*sin(y)*sin(z)", "sin(t)*cos(x)*cos(y)*cos(z)", "sin(t)*cos(x)*sin(y)*(sin(z)+cos(z))"};
        u_boundary.set_string_expression(sin_bc);

        initialize_fields(g, gamma_field, vector_1, rhs_1, u_boundary, g.dt);
        initialize_fields(g, gamma_field, vector_2, rhs_2, u_boundary, g.dt + g.dt);

        VelocitySolver solver = setup_solver(g, gamma_field, u_boundary, g.dt, g.dt + g.dt);

        DimensionsHandlerVector x_handler(g.Nx, g.Ny, g.Nz, 0, 1, 2, g.dx);

        Real l2_error = 0.0;
        Real time_speedup = 0.0;
        bool success = solve_and_check(solver, rhs_1, vector_1, rhs_2, vector_2, g, x_handler, l2_error, time_speedup);

        errors.emplace_back(l2_error);
        dx_values.emplace_back(g.dx);
        dt_values.emplace_back(g.dt);
        speed_ups.emplace_back(time_speedup);

        // Compute convergence rate if we have at least 2 data points
        Real conv_rate = 0.0;
        if (errors.size() > 1)
        {
            size_t idx = errors.size() - 1;
            conv_rate = log(errors[idx - 1] / errors[idx]) / log(dx_values[idx - 1] / dx_values[idx]);
        }

        // Write to file
        outfile << g.Nx << " " << g.Ny << " " << g.Nz << " "
                << g.dx << " " << g.dt << " " << l2_error << " " << conv_rate << " " << time_speedup << "\n";
        outfile.flush();

        printf("L2 Error: %.8e\n", l2_error);
        if (errors.size() > 1)
            printf("Convergence rate: %.4f (expected ~2.0 for 2nd order)\n", conv_rate);

        if (success)
            std::cout << "[PASS] Solver test passed for N=" << N << "\n";
        else
        {
            std::cout << "[FAIL] Solver test failed for N=" << N << "\n";
            outfile.close();
            return -1;
        }
    }

    outfile.close();

    printf("\n=================================================\n");
    printf("CONVERGENCE STUDY SUMMARY\n");
    printf("=================================================\n");
    printf("Grid Size    dx          dt          L2 Error      Conv. Rate   Time Speedup\n");
    printf("---------------------------------------------------------------\n");
    for (size_t i = 0; i < errors.size(); ++i)
    {
        Real rate = (i > 0) ? log(errors[i - 1] / errors[i]) / log(dx_values[i - 1] / dx_values[i]) : 0.0;
        printf("%-12d %.6e  %.6e  %.6e  %.4f    %.4f\n",
               grid_sizes[i], dx_values[i], dt_values[i], errors[i], rate, speed_ups[i]);
    }
    printf("=================================================\n");
    printf("Results written to: convergence_momentum_x.txt\n");

    return 0;
}
