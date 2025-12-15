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
void initialize_vector_field(const Grid &g,
                             VectorVariable &vector,
                             BoundaryFunctions &u_boundary,
                             Real t)
{
    (void)u_boundary; // Unused parameter
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

                    if (comp == 0)
                        vector.set(comp, i, j, k) = sin(t) * sin(x) * sin(y) * sin(z);
                    else if (comp == 1)
                        vector.set(comp, i, j, k) = sin(t) * cos(x) * cos(y) * cos(z);
                    else
                        vector.set(comp, i, j, k) = sin(t) * cos(x) * sin(y) * (sin(z) + cos(z));
                }
}

void initialize_fields(const Grid &g,
                       ScalarVariable &gamma_field,
                       VectorVariable &vector,
                       VectorVariable &rhs,
                       BoundaryFunctions &u_boundary,
                       Real t,
                       Real nu)
{
    initialize_vector_field(g, vector, u_boundary, t);
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

void construct_f_function(VectorVariable &f_function,
                          ScalarVariable &gamma_field,
                          const Grid &g,
                          Real t_half,
                          Real nu)
{
    for (int comp = 0; comp < 3; ++comp)
        for (Dim k = 0; k < g.Nz; ++k)
            for (Dim j = 0; j < g.Ny; ++j)
                for (Dim i = 0; i < g.Nx; ++i)
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

                    Real f_val;

                    if (comp == 0)
                    {
                        Real ut = std::cos(t_half) * std::sin(x) * std::sin(y) * std::sin(z);
                        Real uxx = -std::sin(t_half) * std::sin(x) * std::sin(y) * std::sin(z);
                        f_val = ut - nu * uxx;
                    }
                    else if (comp == 1)
                    {
                        Real ut = std::cos(t_half) * std::cos(x) * std::cos(y) * std::cos(z);
                        Real uxx = -std::sin(t_half) * std::cos(x) * std::cos(y) * std::cos(z);
                        f_val = ut - nu * uxx;
                    }
                    else
                    {
                        Real ut = std::cos(t_half) * std::cos(x) * std::sin(y) * (std::sin(z) + std::cos(z));
                        Real uxx = -std::sin(t_half) * std::cos(x) * std::sin(y) * (std::sin(z) + std::cos(z));
                        f_val = ut - nu * uxx;
                    }

                    f_function.set(comp, i, j, k) = f_val;
                }
}

void solve(VelocitySolver &solver,
           VectorVariable &u_n,
           VectorVariable &u_np1,
           VectorVariable &f_function,
           ScalarVariable &gamma_field,
           const Grid &g,
           const DimensionsHandlerVector &x_handler,
           Real t_start,
           Real T_final,
           Real nu,
           Real beta,
           Real &l2_error,
           Real &time,
           bool parallel = false)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    VectorVariable delta_u(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    VectorVariable rhs(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    VectorVariable u_true(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    VectorVariable rhs_dummy(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);

    Real t_current = t_start;
    int nsteps = int((T_final - t_start) / g.dt);

    for (int n = 0; n < nsteps; ++n)
    {

        Real t_half = t_current + 0.5 * g.dt;
        solver.set_t(t_half);

        /* ------------------------------
         * Build forcing f^{n+1/2}
         * ------------------------------ */
        construct_f_function(f_function, gamma_field, g, t_half, nu);

        /* ------------------------------
         * RHS:
         * dt*f^{n+1/2} + dt*gamma*Dxx(u_n)
         * ------------------------------ */
        for (int comp = 0; comp < 3; ++comp)
            for (Dim k = 0; k < g.Nz; ++k)
                for (Dim j = 0; j < g.Ny; ++j)
                    for (Dim i = 0; i < g.Nx; ++i)
                    {
                        Real Dxx_u = u_n.second_derivative(comp, 0, i, j, k);
                        rhs.set(comp, i, j, k) =
                            g.dt * f_function.value(comp, i, j, k) + g.dt * (nu)*Dxx_u;
                    }

        /* ------------------------------
         * Solve:
         * (I - dt/2*gamma*Dxx) delta_u = rhs
         * ------------------------------ */
        delta_u.set_all(0.0);
        solver.solve<0>(rhs, delta_u, x_handler, parallel);

        /* ------------------------------
         * Update solution
         * ------------------------------ */
        u_np1 = u_n + delta_u;
        u_n = u_np1;
        t_current += g.dt;

        /* ------------------------------
         * Monitor error
         * ------------------------------ */
        if (n % 100 == 0 || n == nsteps - 1)
        {
            initialize_fields(g,
                              solver.gamma_field,
                              u_true,
                              rhs_dummy,
                              solver.u_boundary,
                              t_current,
                              nu);

            std::cout << "Step " << n
                      << " | L2 Error = "
                      << compute_L2_error(u_true, u_n, g)
                      << std::endl;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration<Real>(t1 - t0).count();

    initialize_fields(g,
                      solver.gamma_field,
                      u_true,
                      rhs_dummy,
                      solver.u_boundary,
                      t_current,
                      nu);

    l2_error = compute_L2_error(u_true, u_n, g);
}

// ===============================================================
int main()
{
    const Real two_pi = 6.283185307179586;

    std::ofstream outfile("test_heat_equation.txt");
    outfile << "# Convergence study for x-direction heat solver\n";
    outfile << "# Nx Ny Nz dx dt L2_error convergence_rate\n";
    outfile << std::scientific << std::setprecision(8);

    std::vector<Dim> grid_sizes = {20, 40, 80};
    std::vector<Real> errors;
    std::vector<Real> dx_values;
    std::vector<Real> dt_values;

    Dim iteration = 0;

    for (Dim N : grid_sizes)
    {
        // Keep final T fixed; scale dt ~ dx^2 so temporal error (O(dt^2)) stays well below spatial (O(dx^2))
        Real base_dx = two_pi / (N - 0.5);
        Real dt_test = 0.001 * base_dx; // parabolic scaling
        iteration++;

        Grid g = setup_grid(two_pi, two_pi, two_pi, N, N, N, dt_test);

        printf("\n=================================================\n");
        printf("Grid: N=%d  dx=%e  dt=%e\n", N, g.dx, g.dt);
        printf("=================================================\n");

        /* ---------------------------------------------
         * Parameters
         * --------------------------------------------- */
        Real nu = 0.1;
        Real beta = 1.0;

        /* ---------------------------------------------
         * Gamma field (dt*nu/2)
         * --------------------------------------------- */
        ScalarVariable gamma_field(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
        gamma_field.set_all(nu * g.dt / 2.0);

        /* ---------------------------------------------
         * Solution variables
         * --------------------------------------------- */
        VectorVariable u_n(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
        VectorVariable u_np1(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
        VectorVariable f_fun(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);

        /* ---------------------------------------------
         * Boundary conditions
         * --------------------------------------------- */
        BoundaryFunctions u_boundary;
        std::vector<std::string> sin_bc =
            {
                "sin(t)*sin(x)*sin(y)*sin(z)",
                "sin(t)*cos(x)*cos(y)*cos(z)",
                "sin(t)*cos(x)*sin(y)*(sin(z)+cos(z))"};
        u_boundary.set_string_expression(sin_bc);

        /* ---------------------------------------------
         * Initial condition at t = dt
         * --------------------------------------------- */
        VectorVariable rhs_dummy(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
        initialize_fields(g, gamma_field, u_n, rhs_dummy, u_boundary, g.dt, nu);
        initialize_fields(g, gamma_field, u_np1, rhs_dummy, u_boundary, g.dt + g.dt, nu);

        /* ---------------------------------------------
         * Solver setup
         * --------------------------------------------- */
        VelocitySolver solver =
            setup_solver(g, gamma_field, u_boundary, g.dt, g.dt);

        DimensionsHandlerVector x_handler(
            g.Nx, g.Ny, g.Nz,
            0, 1, 2,
            g.dx);

        /* ---------------------------------------------
         * Time integration with constant physical end time
         * dt scales with dx^2 to keep temporal error < spatial error
         * --------------------------------------------- */
        Real T_final = 0.005; // Larger T to get measurable discretization errors
        Real l2_error = 0.0;
        Real runtime = 0.0;

        solve(solver,
              u_n,
              u_np1,
              f_fun,
              gamma_field,
              g,
              x_handler,
              g.dt,
              T_final,
              nu,
              beta,
              l2_error,
              runtime,
              true);

        /* ---------------------------------------------
         * Store results
         * --------------------------------------------- */
        errors.push_back(l2_error);
        dx_values.push_back(g.dx);
        dt_values.push_back(g.dt);

        Real conv_rate = 0.0;
        if (errors.size() > 1)
        {
            size_t k = errors.size() - 1;
            conv_rate =
                std::log(errors[k - 1] / errors[k]) /
                std::log(dx_values[k - 1] / dx_values[k]);
        }

        outfile << g.Nx << " " << g.Ny << " " << g.Nz << " "
                << g.dx << " " << g.dt << " "
                << l2_error << " " << conv_rate << "\n";

        printf("L2 Error = %.8e\n", l2_error);
    }

    outfile.close();

    printf("\n=================================================\n");
    printf("CONVERGENCE STUDY SUMMARY\n");
    printf("=================================================\n");
    printf("Grid    dx          dt          L2 Error      Rate\n");
    printf("-------------------------------------------------\n");

    for (size_t i = 0; i < errors.size(); ++i)
    {
        Real rate = (i > 0)
                        ? std::log(errors[i - 1] / errors[i]) /
                              std::log(dx_values[i - 1] / dx_values[i])
                        : 0.0;

        printf("%-6d %.6e  %.6e  %.6e  %.3f\n",
               grid_sizes[i],
               dx_values[i],
               dt_values[i],
               errors[i],
               rate);
    }

    printf("=================================================\n");
    printf("Results written to test_heat_equation.txt\n");

    return 0;
}
