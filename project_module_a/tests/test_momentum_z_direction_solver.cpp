#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
#include <chrono>
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
// 2. Apply known Dirichlet faces on vector [NEVER CALLED]
// ===============================================================
void apply_known_faces(VectorVariable &vector,
                       BoundaryFunctions &u_boundary,
                       const Grid &g,
                       ScalarVariable &gamma_field,
                       Real t = 0.0)
{
    (void)gamma_field; // Unused parameter
    Real Lx = g.dx * (g.Nx - 0.5);
    Real Ly = g.dy * (g.Ny - 0.5);
    Real Lz = g.dz * (g.Nz - 0.5);

    // X-direction faces
    for (Dim j = 0; j < g.Ny; ++j)
        for (Dim k = 0; k < g.Nz; ++k)
        {
            vector.set(0, 0, j, k) = u_boundary.value<0>(0.0, j * g.dy, k * g.dz, t);
            vector.set(0, g.Nx - 1, j, k) = u_boundary.value<0>(Lx, j * g.dy, k * g.dz, t);

            vector.set(1, 0, j, k) = u_boundary.value<1>(0.0, 0.5 * g.dy + j * g.dy, k * g.dz, t);
            vector.set(1, g.Nx - 1, j, k) = u_boundary.value<1>(Lx, 0.5 * g.dy + j * g.dy, k * g.dz, t);

            vector.set(2, 0, j, k) = u_boundary.value<2>(0.0, j * g.dy, 0.5 * g.dz + k * g.dz, t);
            vector.set(2, g.Nx - 1, j, k) = u_boundary.value<2>(Lx, j * g.dy, 0.5 * g.dz + k * g.dz, t);
        }

    // Y-direction faces
    for (Dim i = 0; i < g.Nx; ++i)
        for (Dim k = 0; k < g.Nz; ++k)
        {
            vector.set(1, i, 0, k) = u_boundary.value<1>(i * g.dx, 0.0, k * g.dz, t);
            vector.set(1, i, g.Ny - 1, k) = u_boundary.value<1>(i * g.dx, Ly, k * g.dz, t);

            vector.set(0, i, 0, k) = u_boundary.value<0>(0.5 * g.dx + i * g.dx, 0.0, k * g.dz, t);
            vector.set(0, i, g.Ny - 1, k) = u_boundary.value<0>(0.5 * g.dx + i * g.dx, Ly, k * g.dz, t);

            vector.set(2, i, 0, k) = u_boundary.value<2>(i * g.dx, 0.0, 0.5 * g.dz + k * g.dz, t);
            vector.set(2, i, g.Ny - 1, k) = u_boundary.value<2>(i * g.dx, Ly, 0.5 * g.dz + k * g.dz, t);
        }

    // Z-direction faces
    for (Dim i = 0; i < g.Nx; ++i)
        for (Dim j = 0; j < g.Ny; ++j)
        {
            vector.set(2, i, j, 0) = u_boundary.value<2>(i * g.dx, j * g.dy, 0.0, t);
            vector.set(2, i, j, g.Nz - 1) = u_boundary.value<2>(i * g.dx, j * g.dy, Lz, t);

            vector.set(0, i, j, 0) = u_boundary.value<0>(0.5 * g.dx + i * g.dx, j * g.dy, 0.0, t);
            vector.set(0, i, j, g.Nz - 1) = u_boundary.value<0>(0.5 * g.dx + i * g.dx, j * g.dy, Lz, t);

            vector.set(1, i, j, 0) = u_boundary.value<1>(i * g.dx, 0.5 * g.dy + j * g.dy, 0.0, t);
            vector.set(1, i, j, g.Nz - 1) = u_boundary.value<1>(i * g.dx, 0.5 * g.dy + j * g.dy, Lz, t);
        }
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

                    vector.set(comp, i, j, k) = sin(x) * sin(y) * sin(z) * sin(t);
                }

    // Apply known faces
    // apply_known_faces(vector, u_boundary, g, gamma_field, 0.0);

    // Build RHS = (I - gamma * Dzz) * vector
    for (int comp = 0; comp < 3; ++comp)
        for (Dim k = 0; k < g.Nz; ++k)
            for (Dim j = 0; j < g.Ny; ++j)
                for (Dim i = 0; i < g.Nx; ++i)
                {
                    Real val = vector.value(comp, i, j, k);
                    Real gamma_val = gamma_field.get(i, j, k);
                    Real sd = vector.second_derivative(comp, 2, i, j, k);
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

auto define_stride_z(const Grid &g)
{
    return [=](Dim i, Dim j)
    { return i + j * g.Nx; };
}

// ===============================================================
// 5. Compute L2 error between expected and computed solutions
// ===============================================================
Real compute_L2_error(const VectorVariable &expected, const VectorVariable &computed, const Grid &g)
{
    Real error_sq = 0.0;
    Real dV = g.dx * g.dy * g.dz;

    for (int comp = 0; comp < 3; ++comp)
        for (Dim k = 0; k < g.Nz; ++k)
            for (Dim j = 0; j < g.Ny; ++j)
                for (Dim i = 0; i < g.Nx; ++i)
                {
                    Real diff = expected.value(comp, i, j, k) - computed.value(comp, i, j, k);
                    error_sq += diff * diff * dV;
                }

    return std::sqrt(error_sq);
}

// ===============================================================
// 6. Solve and check solution
// ===============================================================
bool solve_and_check(VelocitySolver &solver, VectorVariable &rhs_1,
                     VectorVariable &vector_1, VectorVariable &rhs_2,
                     VectorVariable &vector_2, const Grid &g,
                     const DimensionsHandlerVector &z_handler,
                     Real &l2_error, Real &time_speedup)
{
    VectorVariable rhs_delta(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    VectorVariable vector_delta(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);

    rhs_delta = rhs_2 - rhs_1;
    vector_delta = vector_2 - vector_1;

    // PARALLEL solve
    VectorVariable computed_par(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    computed_par.set_all(0.0);
    auto tpar0 = std::chrono::high_resolution_clock::now();
    solver.solve<2>(rhs_delta, computed_par, z_handler, true);
    auto tpar1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dpar = tpar1 - tpar0;

    l2_error = compute_L2_error(vector_delta, computed_par, g);

    Real tolerance = std::max(5e-5, 0.1 * g.dx * g.dx);

    for (int comp = 0; comp < 3; ++comp)
        for (Dim k = 0; k < g.Nz; ++k)
            for (Dim j = 0; j < g.Ny; ++j)
                for (Dim i = 0; i < g.Nx; ++i)
                    if (std::abs(vector_delta.value(comp, i, j, k) - computed_par.value(comp, i, j, k)) > tolerance)
                    {
                        printf("Mismatch comp=%d (i,j,k)=(%d,%d,%d): expected %f, got %f\n",
                               comp, i, j, k, vector_delta.value(comp, i, j, k), computed_par.value(comp, i, j, k));
                        return false;
                    }

    // SERIAL solve for timing
    VectorVariable computed_ser(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    computed_ser.set_all(0.0);
    auto tser0 = std::chrono::high_resolution_clock::now();
    solver.solve<2>(rhs_delta, computed_ser, z_handler, false);
    auto tser1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dser = tser1 - tser0;

    time_speedup = dser.count() / dpar.count();

    return true;
}

// ===============================================================
// 7. Main function with convergence study
// ===============================================================
int main()
{
    const Real two_pi = 2.0 * 3.141592653589793;
    std::vector<Dim> grid_sizes = {20, 40, 80, 160};
    std::vector<Real> errors;
    std::vector<Real> speed_ups;
    std::vector<Real> dx_values;
    std::vector<Real> dt_values;

    printf("=================================================\n");
    printf("Z-DIRECTION MOMENTUM SPLITTING CONVERGENCE STUDY\n");
    printf("=================================================\n\n");

    std::ofstream outfile("convergence_momentum_z.txt");
    outfile << "# Nx Ny Nz dx dt L2_error convergence_rate\n";
    for (Dim N : grid_sizes)
    {
        // dt ~ O(dx^2) to keep temporal error negligible
        Real dx_nominal = two_pi / (N - 0.5);
        Real dt = 0.001 * dx_nominal;
        Grid g = setup_grid(two_pi, two_pi, two_pi, N, N, N, dt);

        printf("=================================================\n");
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
        std::vector<std::string> sin_bc = {"sin(x)*sin(y)*sin(z)*sin(t)", "sin(x)*sin(y)*sin(z)*sin(t)", "sin(x)*sin(y)*sin(z)*sin(t)"};
        u_boundary.set_string_expression(sin_bc);

        initialize_fields(g, gamma_field, vector_1, rhs_1, u_boundary, g.dt);
        initialize_fields(g, gamma_field, vector_2, rhs_2, u_boundary, g.dt + g.dt);

        VelocitySolver solver = setup_solver(g, gamma_field, u_boundary, g.dt, g.dt + g.dt);

        DimensionsHandlerVector z_handler(g.Nz, g.Nx, g.Ny, 2, 0, 1, g.dz);

        Real l2_error = 0.0;
        Real time_speedup = 0.0;
        bool success = solve_and_check(solver, rhs_1, vector_1, rhs_2, vector_2, g, z_handler, l2_error, time_speedup);

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
        speed_ups.push_back(time_speedup);

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
    printf("Grid Size    dx          dt          L2 Error      Conv. Rate   Time Speedup\n");
    printf("---------------------------------------------------------------\n");
    for (size_t i = 0; i < errors.size(); ++i)
    {
        Real rate = (i > 0) ? log(errors[i - 1] / errors[i]) / log(dx_values[i - 1] / dx_values[i]) : 0.0;
        printf("%-12d %.6e  %.6e  %.6e  %.4f    %.4f\n",
               grid_sizes[i], dx_values[i], dt_values[i], errors[i], rate, speed_ups[i]);
    }
    printf("=================================================\n");
    printf("Results written to: convergence_momentum_z.txt\n");

    return 0;
}
