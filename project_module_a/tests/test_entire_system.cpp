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

void compute_rhs_field(const Grid &g,
                       ScalarVariable &gamma_field,
                       VectorVariable &vector,
                       VectorVariable &rhs,
                       Real t,
                       Real nu)
{
    (void)t; // Unused parameter
    // Build RHS based on vector field
    for (int comp = 0; comp < 3; ++comp)
        for (Dim k = 0; k < g.Nz; ++k)
            for (Dim j = 0; j < g.Ny; ++j)
                for (Dim i = 0; i < g.Nx; ++i)
                {
                    Real gamma_val = gamma_field.get(i, j, k);
                    Real sd = vector.second_derivative(comp, 0, i, j, k);
                    Real rhs_val = vector.value(comp, i, j, k) - nu * gamma_val * sd;

                    rhs.set(comp, i, j, k) = rhs_val;
                }
}

void initialize_velocity_fields(const Grid &g,
                                ScalarVariable &gamma_field,
                                VectorVariable &vector,
                                VectorVariable &rhs,
                                BoundaryFunctions &u_boundary,
                                Real t,
                                Real nu)
{
    initialize_vector_field(g, vector, u_boundary, t);
    compute_rhs_field(g, gamma_field, vector, rhs, t, nu);
}

void initialize_scalar_fields(const Grid &g,
                              ScalarVariable &scalar,
                              Real t)
{
    for (int k = 0; k < g.Nz; ++k)
        for (int j = 0; j < g.Ny; ++j)
            for (int i = 0; i < g.Nx; ++i)
            {
                Real x, y, z;
                x = i * g.dx;
                y = j * g.dy;
                z = k * g.dz;

                scalar.set(i, j, k) = cos(x) * cos(y) * cos(z) * sin(t);
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

// ===============================================================
// 4. Setup Solver & Strides
// ===============================================================
PressureSolver setup_solver(const Grid &g, BoundaryFunctions &p_boundary, Real t)
{
    PressureSolver solver(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz, t, p_boundary);
    solver.p_boundary = p_boundary;
    return solver;
}

// ===============================================================
// 6. Solve and check solution
// ===============================================================
Real vector_compute_L2_error(VectorVariable &expected, VectorVariable &computed, const Grid &g)
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

Real scalar_compute_L2_error(ScalarVariable &expected, ScalarVariable &computed, const Grid &g)
{
    Real error = 0.0;
    for (Dim k = 0; k < g.Nz; ++k)
        for (Dim j = 0; j < g.Ny; ++j)
            for (Dim i = 0; i < g.Nx; ++i)
            {
                Real diff = expected.get(i, j, k) - computed.get(i, j, k);
                error += diff * diff;
            }
    return sqrt(error * g.dx * g.dy * g.dz);
}

void construct_g_function(VectorVariable &g_function,
                          const Grid &g,
                          BoundaryFunctions &u_boundary,
                          Real t,
                          Real nu,
                          Real k_val,
                          VectorVariable &u_n,
                          VectorVariable &eta_n,
                          VectorVariable &zeta_n,
                          ScalarVariable &pressure)
{
    (void)u_boundary; // Unused parameter
    // Compute g = f - (nu/2)*(dxx_eta + dyy_zeta + dzz_u)
    // where f is analytical forcing and second derivatives are numerical
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

                    // Compute analytical pressure gradient
                    // p(x,y,z,t) = cos(x)*cos(y)*cos(z)*sin(t)
                    // grad_p_x = -sin(x)*cos(y)*cos(z)*sin(t)
                    // grad_p_y = -cos(x)*sin(y)*cos(z)*sin(t)
                    // grad_p_z = -cos(x)*cos(y)*sin(z)*sin(t)
                    Real grad_p_analytical;
                    Real grad_p;
                    if (comp == 0)
                    {
                        grad_p_analytical = -sin(x) * cos(y) * cos(z) * sin(t);
                        grad_p = pressure.getGradient_x(i, j, k);
                    }
                    else if (comp == 1)
                    {
                        grad_p_analytical = -cos(x) * sin(y) * cos(z) * sin(t);
                        grad_p = pressure.getGradient_y(i, j, k);
                    }
                    else
                    {
                        grad_p_analytical = -cos(x) * cos(y) * sin(z) * sin(t);
                        grad_p = pressure.getGradient_z(i, j, k);
                    }

                    // Compute analytical forcing term f = du/dt - nu*Laplacian(u) + (nu/k)*u - grad_p
                    Real f_val;
                    if (comp == 0)
                    {
                        Real u_t = cos(t) * sin(x) * sin(y) * sin(z);
                        Real laplacian_u = -3.0 * sin(t) * sin(x) * sin(y) * sin(z);
                        f_val = u_t - nu * laplacian_u + (nu / (2.0 * k_val)) * (sin(t) * sin(x) * sin(y) * sin(z)) - grad_p_analytical;
                    }
                    else if (comp == 1)
                    {
                        Real v_t = cos(t) * cos(x) * cos(y) * cos(z);
                        Real laplacian_v = -3.0 * sin(t) * cos(x) * cos(y) * cos(z);
                        f_val = v_t - nu * laplacian_v + (nu / (2.0 * k_val)) * (sin(t) * cos(x) * cos(y) * cos(z)) - grad_p_analytical;
                    }
                    else
                    {
                        Real w_t = cos(t) * cos(x) * sin(y) * (sin(z) + cos(z));
                        Real laplacian_w = -3.0 * sin(t) * cos(x) * sin(y) * (sin(z) + cos(z));
                        f_val = w_t - nu * laplacian_w + (nu / (2.0 * k_val)) * (sin(t) * cos(x) * sin(y) * (sin(z) + cos(z))) - grad_p_analytical;
                    }

                    // Compute numerical second derivatives from current solution
                    Real dxx_eta = eta_n.second_derivative(comp, 0, i, j, k);
                    Real dyy_zeta = zeta_n.second_derivative(comp, 1, i, j, k);
                    Real dzz_u = u_n.second_derivative(comp, 2, i, j, k);

                    // g = f - (nu/2)*(dxx + dyy + dzz) - (nu/(2k))*u
                    Real g_val = f_val - (nu / (2.0 * k_val)) * u_n.value(comp, i, j, k) - (nu / 2.0) * (dxx_eta + dyy_zeta + dzz_u) - grad_p;

                    g_function.set(comp, i, j, k) = g_val;
                }
}

void compute_xi_function(VectorVariable &xi_function, VectorVariable &g_function, const Grid &g, VectorVariable &u_1, Real beta)
{
    for (int comp = 0; comp < 3; ++comp)
        for (Dim k = 0; k < g.Nz; ++k)
            for (Dim j = 0; j < g.Ny; ++j)
                for (Dim i = 0; i < g.Nx; ++i)
                {
                    Real val = u_1.value(comp, i, j, k) + (g.dt / Real(beta)) * g_function.value(comp, i, j, k);
                    xi_function.set(comp, i, j, k) = val;
                }
}

void compute_fist_rhs_pressure(ScalarVariable &rhs_pressure,
                               VectorVariable &u_1, const Grid &g)
{
    for (Dim k = 1; k < g.Nz; ++k)
    {
        for (Dim j = 1; j < g.Ny; ++j)
        {
            for (Dim i = 1; i < g.Nx; ++i)
            {

                // Compute RHS for pressure Poisson equation

                rhs_pressure.set(i, j, k) =
                    -(Real(1.0) / g.dt) *
                    u_1.divergence(i, j, k);
            }
        }
    }
}

void solve(VelocitySolver &solver_mom,
           PressureSolver &solver_p,
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
           Real k_val,
           Real &l2_error_velocity,
           Real &l2_error_pressure,
           Real &time,

           bool parallel = false)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    VectorVariable rhs_mom(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    VectorVariable temp_solution(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    VectorVariable u_true_next(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);

    VectorVariable rhs_true_next(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    VectorVariable u_n(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    VectorVariable eta_n(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    VectorVariable zeta_n(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);

    ScalarVariable pressure(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    ScalarVariable p_true_next(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    ScalarVariable rhs_press(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    ScalarVariable psi(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    ScalarVariable phi(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    ScalarVariable other_phi(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);

    // Initialize u_n, eta_n, zeta_n from input (at t_n)
    u_n = u_1;
    eta_n = eta_1;
    zeta_n = zeta_1;
    initialize_scalar_fields(g, pressure, t_n_plus_1_start);

    // Time-stepping loop from t_n_plus_1_start to T_final
    int num_steps = (int)((T_final - t_n_plus_1_start) / g.dt) + 1;
    Real t_current = t_n_plus_1_start;

    for (int step = 0; step < num_steps; ++step)
    {
        // Update solver_mom time to current time step
        solver_mom.set_t(t_current);
        solver_p.set_t(t_current);

        // Compute true solution at current time for error computation
        initialize_velocity_fields(g, solver_mom.gamma_field, u_true_next, rhs_true_next, solver_mom.u_boundary, t_current, nu);
        initialize_scalar_fields(g, p_true_next, t_current);

        // Construct g_function: g = f - (nu/2)*(dxx_eta + dyy_zeta + dzz_u)
        construct_g_function(g_function, g, solver_mom.u_boundary, t_current, nu, k_val, u_n, eta_n, zeta_n, pressure);
        compute_xi_function(xi_function, g_function, g, u_n, beta);

        // X-direction splitting: solve for eta_{n+1}
        temp_solution.set_all(0.0f);
        rhs_mom = xi_function - eta_n;
        solver_mom.solve<0>(rhs_mom, temp_solution, x_handler, parallel);
        eta_2 = temp_solution + eta_n;

        // Y-direction splitting: solve for zeta_{n+1}
        rhs_mom = eta_2 - zeta_n;
        temp_solution.set_all(0.0f);
        solver_mom.solve<1>(rhs_mom, temp_solution, y_handler, parallel);
        zeta_2 = temp_solution + zeta_n;

        // Z-direction splitting: solve for u_{n+1}
        rhs_mom = zeta_2 - u_n;
        temp_solution.set_all(0.0f);
        solver_mom.solve<2>(rhs_mom, temp_solution, z_handler, parallel);
        VectorVariable u_n_plus_1(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
        u_n_plus_1 = temp_solution + u_n;

        compute_fist_rhs_pressure(rhs_press, u_n_plus_1, g);
        psi.set_all(0.0f);
        solver_p.solve_pressure<0>(rhs_press, psi, x_handler);

        phi.set_all(0.0f);
        solver_p.solve_pressure<1>(psi, phi, y_handler);

        other_phi.set_all(0.0f);
        solver_p.solve_pressure<2>(phi, other_phi, z_handler);
        pressure += other_phi;

        // Update for next iteration
        u_n = u_n_plus_1;
        eta_n = eta_2;
        zeta_n = zeta_2;

        // Advance time
        t_current += g.dt;

        if (step % 10 == 0 || step == num_steps - 1)
        {
            std::cout << " Velocity Error at step " << step + 1 << ": " << vector_compute_L2_error(u_true_next, u_n, g) << std::scientific << std::setprecision(8) << std::endl;
            std::cout << " Pressure Error at step " << step + 1 << ": " << scalar_compute_L2_error(p_true_next, pressure, g) << std::scientific << std::setprecision(8) << std::endl;
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration<Real>(end_time - start_time).count();

    // Compute L2 error at final time only
    initialize_velocity_fields(g, solver_mom.gamma_field, u_true_next, rhs_true_next, solver_mom.u_boundary, t_current - g.dt, nu);
    l2_error_velocity = vector_compute_L2_error(u_true_next, u_n, g);
    l2_error_pressure = scalar_compute_L2_error(p_true_next, pressure, g);
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
    std::vector<Dim> grid_sizes = {40, 80, 160};
    std::vector<Real> l2_errors_velocity;
    std::vector<Real> l2_errors_pressure;
    std::vector<Real> speed_ups;
    std::vector<Real> dx_values;
    std::vector<Real> dt_values;
    /*
    Real first_base_dx = two_pi / (grid_sizes[0] - 0.5);
    Real first_dt = 0.001 * first_base_dx;
    */

    for (Dim N : grid_sizes)
    {
        // Scale dt proportionally with dx for second-order spatial discretization
        // This keeps the temporal error smaller than spatial error
        Real base_dx = two_pi / (N - 0.5);
        Real dt_test = 0.001 * base_dx;
        Grid g = setup_grid(two_pi, two_pi, two_pi, N, N, N, dt_test);

        printf("\n=================================================\n");
        printf("Grid: Nx=%d, Ny=%d, Nz=%d, dx=%f, dy=%f, dz=%f, dt=%f\n",
               g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz, g.dt);
        printf("=================================================\n");

        ScalarVariable gamma_field(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
        Real nu = 0.1f;
        Real beta = 1.0f;
        Real k_val = 1.0f;
        gamma_field.set_all(((g.dt * nu) / (Real(2.0) * beta)));

        /*
        MOMENTUM EQUATION
        */
        VectorVariable u_1_sequential(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
        VectorVariable eta_1_sequential(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
        VectorVariable zeta_1_sequential(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
        VectorVariable rhs_1_sequential(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);

        VectorVariable eta_2_sequential(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
        VectorVariable zeta_2_sequential(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);

        VectorVariable g_function_sequential(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
        VectorVariable xi_function_sequential(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);

        VectorVariable u_1_parallel(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
        VectorVariable eta_1_parallel(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
        VectorVariable zeta_1_parallel(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
        VectorVariable rhs_1_parallel(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);

        VectorVariable eta_2_parallel(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
        VectorVariable zeta_2_parallel(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);

        VectorVariable g_function_parallel(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
        VectorVariable xi_function_parallel(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);

        BoundaryFunctions u_boundary;
        std::vector<std::string> sin_bc = {"sin(t)*sin(x)*sin(y)*sin(z)", "sin(t)*cos(x)*cos(y)*cos(z)", "sin(t)*cos(x)*sin(y)*(sin(z)+cos(z))"};
        u_boundary.set_string_expression(sin_bc);

        initialize_velocity_fields(g, gamma_field, u_1_sequential, rhs_1_sequential, u_boundary, g.dt, nu);
        initialize_velocity_fields(g, gamma_field, eta_1_sequential, rhs_1_sequential, u_boundary, g.dt, nu);
        initialize_velocity_fields(g, gamma_field, zeta_1_sequential, rhs_1_sequential, u_boundary, g.dt, nu);

        initialize_velocity_fields(g, gamma_field, u_1_parallel, rhs_1_parallel, u_boundary, g.dt, nu);
        initialize_velocity_fields(g, gamma_field, eta_1_parallel, rhs_1_parallel, u_boundary, g.dt, nu);
        initialize_velocity_fields(g, gamma_field, zeta_1_parallel, rhs_1_parallel, u_boundary, g.dt, nu);

        VelocitySolver solver_mom = setup_solver(g, gamma_field, u_boundary, g.dt, g.dt + g.dt);

        /*
        PRESSURE EQUATION
        */

        BoundaryFunctions p_boundary;
        std::vector<std::string> neumann_bc = {
            "0",
            "0",
            "0",
        };

        p_boundary.set_string_expression(neumann_bc);

        PressureSolver solver_p = setup_solver(g, p_boundary, g.dt + g.dt);

        DimensionsHandlerVector x_handler(g.Nx, g.Ny, g.Nz, 0, 1, 2, g.dx);

        DimensionsHandlerVector y_handler(g.Nx, g.Ny, g.Nz, 1, 0, 2, g.dy);

        DimensionsHandlerVector z_handler(g.Nx, g.Ny, g.Nz, 2, 0, 1, g.dz);
        Real T_final = 10 * (g.dt);

        Real l2_error_velocity = 0.0;
        Real l2_error_pressure = 0.0;
        Real time_sequential = 0.0;
        bool parallel = false;

        solve(solver_mom,
              solver_p,
              eta_1_sequential,
              eta_2_sequential,
              zeta_1_sequential,
              zeta_2_sequential,
              u_1_sequential,
              g_function_sequential,
              xi_function_sequential,
              g,
              x_handler,
              y_handler,
              z_handler,
              g.dt + g.dt,
              T_final,
              nu,
              beta,
              k_val,
              l2_error_velocity, l2_error_pressure, time_sequential, parallel);
        l2_errors_velocity.push_back(l2_error_velocity);
        l2_errors_pressure.push_back(l2_error_pressure);
        dx_values.push_back(g.dx);
        dt_values.push_back(g.dt);

        // Compute convergence rate if we have at least 2 data points
        Real conv_rate_velocity = 0.0;
        Real conv_rate_pressure = 0.0;
        if (l2_errors_velocity.size() > 1)
        {
            size_t idx = l2_errors_velocity.size() - 1;
            conv_rate_velocity = log(l2_errors_velocity[idx - 1] / l2_errors_velocity[idx]) / log(dx_values[idx - 1] / dx_values[idx]);
            conv_rate_pressure = log(l2_errors_pressure[idx - 1] / l2_errors_pressure[idx]) / log(dx_values[idx - 1] / dx_values[idx]);
        }

        // Write to file
        outfile << g.Nx << " " << g.Ny << " " << g.Nz << " "
                << g.dx << " " << g.dt << " "
                << l2_errors_velocity.back() << " " << conv_rate_velocity << " "
                << l2_errors_pressure.back() << " " << conv_rate_pressure << "\n";
        outfile.flush();

        printf("L2 Error Velocity: %.8e, L2 Error Pressure: %.8e\n",
               l2_errors_velocity.back(), l2_errors_pressure.back());
    }

    outfile.close();

    printf("\n=================================================\n");
    printf("CONVERGENCE STUDY SUMMARY\n");
    printf("=================================================\n");
    printf("Grid Size    dx          dt          L2 Vel Error  Conv. Rate  L2 Pres Error  Conv. Rate\n");
    printf("-----------------------------------------------------------------------------------------\n");
    for (size_t i = 0; i < l2_errors_velocity.size(); ++i)
    {
        Real rate_vel = (i > 0) ? log(l2_errors_velocity[i - 1] / l2_errors_velocity[i]) / log(dx_values[i - 1] / dx_values[i]) : 0.0;
        Real rate_pres = (i > 0) ? log(l2_errors_pressure[i - 1] / l2_errors_pressure[i]) / log(dx_values[i - 1] / dx_values[i]) : 0.0;
        printf("%-12d %.6e  %.6e  %.6e  %.4f  %.6e  %.4f\n",
               grid_sizes[i], dx_values[i], dt_values[i], l2_errors_velocity[i], rate_vel, l2_errors_pressure[i], rate_pres);
    }
    printf("=================================================\n");
    printf("Results written to: convergence_momentum_x.txt\n");

    return 0;
}
