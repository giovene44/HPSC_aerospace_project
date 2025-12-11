#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
#include "navier_stokes_brinkman.hpp"

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
                       ScalarVariable &scalar,
                       ScalarVariable &rhs,
                       BoundaryFunctions &p_boundary,
                       Real t)
{

    // Initialize interior
    for (int k = 0; k < g.Nz; ++k)
        for (int j = 0; j < g.Ny; ++j)
            for (int i = 0; i < g.Nx; ++i)
            {
                Real x, y, z;
                x = i * g.dx;
                y = j * g.dy;
                z = k * g.dz;

                scalar.set(i, j, k) = sin(z) * sin(t);
            }

    // Build RHS = (I - Dzz) * scalar
    // For Neumann BCs, RHS is computed everywhere; boundary gradients handled in solver
    for (Dim k = 0; k < g.Nz; ++k)
        for (Dim j = 0; j < g.Ny; ++j)
            for (Dim i = 0; i < g.Nx; ++i)
            {
                Real val = scalar.get(i, j, k);
                Real sd = scalar.second_derivative(2, i, j, k);
                Real rhs_val = val - sd;
                rhs.set(i, j, k) = rhs_val;
            }
}

// ===============================================================
// 4. Setup Solver & Strides
// ===============================================================
PressureSolver setup_solver(const Grid &g, BoundaryFunctions &p_boundary, Real t)
{
    PressureSolver solver(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz, t, p_boundary);
    solver.p_boundary = p_boundary;
    return solver;
}

auto define_stride_z(const Grid &g)
{
    return [=](Dim i, Dim j)
    { return i + j * g.Nx; };
}

// ===============================================================
// 5. Compute L2 error between expected and computed solutions
// ===============================================================
Real compute_L2_error(const ScalarVariable &expected, const ScalarVariable &computed, const Grid &g)
{
    Real error_sq = 0.0;
    Real dV = g.dx * g.dy * g.dz;

    for (Dim k = 0; k < g.Nz; ++k)
        for (Dim j = 0; j < g.Ny; ++j)
            for (Dim i = 0; i < g.Nx; ++i)
            {
                Real diff = expected.get(i, j, k) - computed.get(i, j, k);
                error_sq += diff * diff * dV;
            }

    return std::sqrt(error_sq);
}

// ===============================================================
// 6. Solve and check solution
// ===============================================================
bool solve_and_check(PressureSolver &solver, ScalarVariable &rhs,
                     ScalarVariable &scalar, const Grid &g,
                     DimensionsHandlerVector &z_handler,
                     Real &l2_error)
{

    ScalarVariable computed_sol(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    computed_sol.set_all(0.0);

    solver.solve_pressure<2>(rhs, computed_sol, z_handler);

    l2_error = compute_L2_error(scalar, computed_sol, g);

    Real tolerance = std::max(1e-3, 0.1 * g.dx * g.dx);
    bool res = true;

    for (Dim j = 0; j < g.Ny; ++j)
        for (Dim k = 0; k < g.Nz; ++k)
            for (Dim i = 0; i < g.Nx; ++i)
                if (std::abs(scalar.get(i, j, k) - computed_sol.get(i, j, k)) > tolerance)
                {
                    printf("Mismatch (i,j,k)=(%d,%d,%d): expected %f, got %f\n",
                           i, j, k, scalar.get(i, j, k), computed_sol.get(i, j, k));
                    res = false;
                }

    return res;
}

// ===============================================================
// 7. Main function with convergence study
// ===============================================================
int main()
{
    constexpr Real two_pi = 2.0 * M_PI;
    std::vector<Dim> grid_sizes = {10, 20, 40, 80, 100};
    std::vector<Real> errors;
    std::vector<Real> dx_values;
    std::vector<Real> dt_values;

    printf("=================================================\n");
    printf("Z-DIRECTION PRESSURE SPLITTING CONVERGENCE STUDY\n");
    printf("=================================================\n\n");

    std::ofstream outfile("convergence_pressure_z.txt");
    outfile << "# Nx Ny Nz dx dt L2_error convergence_rate\n";

    for (Dim N : grid_sizes)
    {
        Real dt = 0.01 * (two_pi / (N - 0.5));
        Grid g = setup_grid(two_pi, two_pi, two_pi, N, N, N, dt);

        printf("=================================================\n");
        printf("Grid: Nx=%d, Ny=%d, Nz=%d, dx=%f, dy=%f, dz=%f, dt=%f\n",
               g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz, g.dt);
        printf("=================================================\n");

        ScalarVariable scalar(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
        ScalarVariable rhs(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);

        BoundaryFunctions p_boundary;
        std::vector<std::string> neumann_bc = {
            "0",
            "0",
            "cos(z)*sin(t)"};

        p_boundary.set_string_expression(neumann_bc);

        initialize_fields(g, scalar, rhs, p_boundary, g.dt);
        PressureSolver solver = setup_solver(g, p_boundary, g.dt);

        DimensionsHandlerVector z_handler(g.Nx, g.Ny, g.Nz, 2, 0, 1, g.dz);

        Real l2_error = 0.0;
        bool success = solve_and_check(solver, rhs, scalar, g, z_handler, l2_error);

        printf("L2 Error: %.8e\n", l2_error);

        Real conv_rate = 0.0;
        if (!errors.empty())
        {
            conv_rate = log(errors.back() / l2_error) / log(dx_values.back() / g.dx);
            printf("Convergence rate: %.4f (expected ~2.0 for 2nd order)\n", conv_rate);
        }

        errors.push_back(l2_error);
        dx_values.push_back(g.dx);
        dt_values.push_back(g.dt);

        outfile << g.Nx << " " << g.Ny << " " << g.Nz << " " << g.dx << " " << g.dt << " " << l2_error << " " << conv_rate << "\n";

        if (success)
            printf("[PASS] Solver test passed for N=%d\n", N);
        else
        {
            printf("[FAIL] Solver test failed for N=%d\n", N);
            return -1;
        }
    }

    outfile.close();

    printf("\n=================================================\n");
    printf("CONVERGENCE STUDY SUMMARY\n");
    printf("=================================================\n");
    printf("Grid Size    dx          dt          L2 Error      Conv. Rate\n");
    printf("---------------------------------------------------------------\n");
    for (size_t i = 0; i < errors.size(); ++i)
    {
        Real rate = (i > 0) ? log(errors[i - 1] / errors[i]) / log(dx_values[i - 1] / dx_values[i]) : 0.0;
        printf("%-12d %.6e  %.6e  %.6e  %.4f\n",
               grid_sizes[i], dx_values[i], dt_values[i], errors[i], rate);
    }
    printf("=================================================\n");
    printf("Results written to: convergence_pressure_z.txt\n");

    return 0;
}
