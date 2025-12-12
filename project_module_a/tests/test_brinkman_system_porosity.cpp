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
// 2. Initialize Fields (interior + RHS)
// ===============================================================
void initialize_vector_field(const Grid &g,
                             VectorVariable &vector,
                             BoundaryFunctions &u_boundary,
                             Real t)
{
    // Initialize interior and boundary
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

                    vector.set(comp, i, j, k) = sin(x) * sin(t) * sin(y) * sin(z);

                    // Apply boundary conditions
                    if (i == 0 || i == g.Nx - 1 || j == 0 || j == g.Ny - 1 || k == 0 || k == g.Nz - 1)
                    {
                        if (comp == 0)
                        {
                            vector.set(comp, i, j, k) = u_boundary.value<0>(x, y, z, t);
                        }
                        else if (comp == 1)
                        {
                            vector.set(comp, i, j, k) = u_boundary.value<1>(x, y, z, t);
                        }
                        else
                        {
                            vector.set(comp, i, j, k) = u_boundary.value<2>(x, y, z, t);
                        }
                    }
                }
}

void compute_rhs_field(const Grid &g,
                       ScalarVariable &gamma_field,
                       VectorVariable &vector,
                       VectorVariable &rhs,
                       Real t,
                       Real nu,
                       Real k_permeability)
{
    // Build RHS based on vector field
    // RHS = du/dt - nu*Laplacian(u) + (nu/k)*u
    for (int comp = 0; comp < 3; ++comp)
        for (Dim k = 0; k < g.Nz; ++k)
            for (Dim j = 0; j < g.Ny; ++j)
                for (Dim i = 0; i < g.Nx; ++i)
                {
                    Real x_coord, y_coord, z_coord;
                    if (comp == 0)
                    {
                        x_coord = (i + 0.5) * g.dx;
                        y_coord = j * g.dy;
                        z_coord = k * g.dz;
                    }
                    else if (comp == 1)
                    {
                        x_coord = i * g.dx;
                        y_coord = (j + 0.5) * g.dy;
                        z_coord = k * g.dz;
                    }
                    else
                    {
                        x_coord = i * g.dx;
                        y_coord = j * g.dy;
                        z_coord = (k + 0.5) * g.dz;
                    }

                    Real u_val = vector.value(comp, i, j, k);
                    Real gamma_val = gamma_field.get(i, j, k);
                    Real sd = vector.second_derivative(comp, 0, i, j, k);
                    Real time_der = sin(x_coord) * cos(t) * sin(y_coord) * sin(z_coord);

                    // RHS = du/dt - nu*Laplacian(u) + (nu/k)*u
                    Real rhs_val = time_der - nu * gamma_val * sd + (nu / k_permeability) * u_val;

                    rhs.set(comp, i, j, k) = rhs_val;
                }
}

void initialize_fields(const Grid &g,
                       ScalarVariable &gamma_field,
                       VectorVariable &vector,
                       VectorVariable &rhs,
                       BoundaryFunctions &u_boundary,
                       Real t,
                       Real nu,
                       Real k_permeability)
{
    initialize_vector_field(g, vector, u_boundary, t);
    compute_rhs_field(g, gamma_field, vector, rhs, t, nu, k_permeability);
}

// ===============================================================
// 3. Setup Solver & Strides
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

auto define_stride_y(const Grid &g)
{
    return [=](Dim i, Dim k)
    { return i + k * g.Nx * g.Ny; };
}

auto define_stride_z(const Grid &g)
{
    return [=](Dim i, Dim j)
    { return i + j * g.Nx; };
}

// ===============================================================
// 4. Compute L2 error
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

// ===============================================================
// 5. Construct g_function and xi_function
// ===============================================================
void construct_g_function(VectorVariable &g_function,
                          VectorVariable &rhs_2,
                          VectorVariable &u_1,
                          VectorVariable &eta_1,
                          VectorVariable &zeta_1,
                          const Grid &g,
                          Real nu)
{
    for (int comp = 0; comp < 3; ++comp)
        for (Dim k = 0; k < g.Nz; ++k)
            for (Dim j = 0; j < g.Ny; ++j)
                for (Dim i = 0; i < g.Nx; ++i)
                {
                    Real val = rhs_2.value(comp, i, j, k) +
                               (nu / Real(2.0)) * (u_1.second_derivative(comp, 0, i, j, k) +
                                                   eta_1.second_derivative(comp, 1, i, j, k) +
                                                   zeta_1.second_derivative(comp, 2, i, j, k));
                    g_function.set(comp, i, j, k) = val;
                }
}

void compute_xi_function(VectorVariable &xi_function,
                         VectorVariable &g_function,
                         const Grid &g,
                         VectorVariable &u_1,
                         Real beta)
{
    for (int comp = 0; comp < 3; ++comp)
        for (Dim k = 0; k < g.Nz; ++k)
            for (Dim j = 0; j < g.Ny; ++j)
                for (Dim i = 0; i < g.Nx; ++i)
                {
                    Real val = u_1.value(comp, i, j, k) +
                               (g.dt / Real(beta)) * g_function.value(comp, i, j, k);
                    xi_function.set(comp, i, j, k) = val;
                }
}

// ===============================================================
// 6. Solve Brinkman system with time stepping
// ===============================================================
void solve(VelocitySolver &solver,
           VectorVariable &eta_1,
           VectorVariable &eta_2,
           VectorVariable &zeta_1,
           VectorVariable &zeta_2,
           VectorVariable &u_1,
           VectorVariable &g_function,
           VectorVariable &xi_function,
           const Grid &g,
           const DimensionsHandlerVector &x_handler,
           const DimensionsHandlerVector &y_handler,
           const DimensionsHandlerVector &z_handler,
           Real t_n_plus_1_start,
           Real T_final,
           Real nu,
           Real beta,
           Real k_permeability,
           Real &l2_error)
{
    VectorVariable rhs(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    VectorVariable temp_solution(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    VectorVariable u_true_next(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    VectorVariable rhs_true_next(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    VectorVariable u_n(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    VectorVariable eta_n(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    VectorVariable zeta_n(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);

    // Initialize u_n, eta_n, zeta_n from input (at t_n)
    u_n = u_1;
    eta_n = eta_1;
    zeta_n = zeta_1;

    // Time-stepping loop from t_n_plus_1_start to T_final
    int num_steps = (int)((T_final - t_n_plus_1_start) / g.dt) + 1;
    Real t_current = t_n_plus_1_start;

    for (int step = 0; step < num_steps; ++step)
    {
        // Update solver time to current time step
        solver.set_t(t_current);

        // Compute true solution at current time for RHS computation
        initialize_fields(g, solver.gamma_field, u_true_next, rhs_true_next,
                          solver.u_boundary, t_current, nu, k_permeability);

        // Construct g_function and xi_function for current time step
        construct_g_function(g_function, rhs_true_next, u_n, eta_n, zeta_n, g, nu);
        compute_xi_function(xi_function, g_function, g, u_n, beta);

        // X-direction splitting: solve for eta_{n+1}
        temp_solution.set_all(0.0f);
        rhs = xi_function - eta_n;
        solver.solve<0>(rhs, temp_solution, x_handler);
        eta_2 = temp_solution + eta_n;

        // Y-direction splitting: solve for zeta_{n+1}
        rhs = eta_2 - zeta_n;
        temp_solution.set_all(0.0f);
        solver.solve<1>(rhs, temp_solution, y_handler);
        zeta_2 = temp_solution + zeta_n;

        // Z-direction splitting: solve for u_{n+1}
        rhs = zeta_2 - u_n;
        temp_solution.set_all(0.0f);
        solver.solve<2>(rhs, temp_solution, z_handler);
        VectorVariable u_n_plus_1(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
        u_n_plus_1 = temp_solution + u_n;

        // Update for next iteration
        u_n = u_n_plus_1;
        eta_n = eta_2;
        zeta_n = zeta_2;

        // Advance time
        t_current += g.dt;

        if (step % 10 == 0 || step == num_steps - 1)
        {
            printf("  Step %d/%d, t = %.6f\n", step + 1, num_steps, t_current);
        }
    }

    // Compute L2 error at final time only
    initialize_fields(g, solver.gamma_field, u_true_next, rhs_true_next,
                      solver.u_boundary, t_current - g.dt, nu, k_permeability);
    l2_error = compute_L2_error(u_true_next, u_n, g);
}

// ===============================================================
// 7. Main function
// ===============================================================
int main()
{
    const Real two_pi = 2.0 * 3.141592653589793;

    // Open output file
    std::ofstream outfile("convergence_brinkman_system.txt");
    outfile << "# Convergence study for Brinkman system: du/dt - nu*div(u) + (nu/k)*u = f\n";
    outfile << "# Nx Ny Nz dx dt L2_error convergence_rate\n";
    outfile << std::scientific << std::setprecision(8);

    // Test multiple grid resolutions
    std::vector<Dim> grid_sizes = {10, 20, 40, 80};
    std::vector<Real> errors;
    std::vector<Real> dx_values;
    std::vector<Real> dt_values;

    for (Dim N : grid_sizes)
    {
        // Make dt proportional to dx to avoid time discretization error dominating
        Real dt_test = 0.01 * (two_pi / (N - 0.5)); // dt ~ O(dx)
        Grid g = setup_grid(two_pi, two_pi, two_pi, N, N, N, dt_test);

        printf("\n=================================================\n");
        printf("Grid: Nx=%d, Ny=%d, Nz=%d, dx=%f, dy=%f, dz=%f, dt=%f\n",
               g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz, g.dt);
        printf("=================================================\n");

        ScalarVariable gamma_field(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
        Real nu = 0.1f;
        Real beta = 1.0f;
        Real k_permeability = 1.0f; // Permeability for Brinkman term

        gamma_field.set_all((g.dt * nu) / (Real(2.0) * beta));

        VectorVariable u_1(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
        VectorVariable eta_1(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
        VectorVariable zeta_1(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
        VectorVariable rhs_1(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);

        VectorVariable eta_2(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
        VectorVariable zeta_2(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);

        VectorVariable g_function(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
        VectorVariable xi_function(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);

        BoundaryFunctions u_boundary;
        std::vector<std::string> sin_bc = {
            "sin(x)*sin(t)*sin(y)*sin(z)",
            "sin(x)*sin(t)*sin(y)*sin(z)",
            "sin(x)*sin(t)*sin(y)*sin(z)"};
        u_boundary.set_string_expression(sin_bc);

        initialize_fields(g, gamma_field, u_1, rhs_1, u_boundary, g.dt, nu, k_permeability);
        initialize_fields(g, gamma_field, eta_1, rhs_1, u_boundary, g.dt, nu, k_permeability);
        initialize_fields(g, gamma_field, zeta_1, rhs_1, u_boundary, g.dt, nu, k_permeability);

        VelocitySolver solver = setup_solver(g, gamma_field, u_boundary, g.dt, g.dt + g.dt);

        DimensionsHandlerVector x_handler(g.Nx, g.Ny, g.Nz, 0, 1, 2, g.dx);

        DimensionsHandlerVector y_handler(g.Nx, g.Ny, g.Nz, 1, 0, 2, g.dy);

        DimensionsHandlerVector z_handler(g.Nx, g.Ny, g.Nz, 2, 0, 1, g.dz);
        Real T_final = 10 * g.dt;

        Real l2_error = 0.0;
        solve(solver,
              eta_1,
              eta_2,
              zeta_1,
              zeta_2,
              u_1,
              g_function,
              xi_function,
              g,
              x_handler,
              y_handler,
              z_handler,
              g.dt + g.dt,
              T_final,
              nu,
              beta,
              k_permeability,
              l2_error);

        errors.emplace_back(l2_error);
        dx_values.emplace_back(g.dx);
        dt_values.emplace_back(g.dt);

        // Compute convergence rate if we have at least 2 data points
        Real conv_rate = 0.0;
        if (errors.size() > 1)
        {
            size_t idx = errors.size() - 1;
            conv_rate = log(errors[idx - 1] / errors[idx]) / log(dx_values[idx - 1] / dx_values[idx]);
        }

        // Write to file
        outfile << g.Nx << " " << g.Ny << " " << g.Nz << " "
                << g.dx << " " << g.dt << " " << l2_error << " " << conv_rate << "\n";
        outfile.flush();

        printf("L2 Error: %.8e\n", l2_error);
    }

    outfile.close();

    printf("\n=================================================\n");
    printf("CONVERGENCE STUDY SUMMARY\n");
    printf("=================================================\n");
    printf("Brinkman System: du/dt - nu*div(u) + (nu/k)*u = f\n");
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
    printf("Results written to: convergence_brinkman_system.txt\n");

    return 0;
}
