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
// 2. Apply known Dirichlet faces on vector [NEVER CALLED]
// ===============================================================
void apply_known_faces(VectorVariable &vector,
                       BoundaryFunctions &u_boundary,
                       const Grid &g,
                       ScalarVariable &gamma_field,
                       Real t = 0.0)
{
    Real Lx = g.dx * (g.Nx - 0.5);
    Real Ly = g.dy * (g.Ny - 0.5);
    Real Lz = g.dz * (g.Nz - 0.5);

    // X-direction faces
    for (Dim j = 0; j < g.Ny; ++j)
        for (Dim k = 0; k < g.Nz; ++k)
        {
            vector.set(0, 0, j, k) = u_boundary.value<0>(0.0, j * g.dy, k * g.dz, t);
            vector.set(0, g.Nx - 1, j, k) = u_boundary.value<0>(Lx, j * g.dy, k * g.dz, t);
            /*
            vector.set(1, 0, j, k) = u_boundary.value<1>(0.0, 0.5 * g.dy + j * g.dy, k * g.dz, t);
            Real val = vector.value(1, g.Nx - 1, j, k);
            Real gamma_val = gamma_field.get(g.Nx - 1, j, k);
            Real sd = vector.second_derivative(1, 0, g.Nx - 1, j, k);
            Real rhs_val = val - gamma_val * sd;
            auto a = -gamma_val / (g.dx * g.dx);
            auto b = 1.0 + 2.0 * gamma_val / (g.dx * g.dx);
            auto c = -gamma_val / (g.dx * g.dx);
            vector.set(1, g.Nx - 1, j, k) = ((rhs_val - 2 * c * u_boundary.value<1>(Lx, 0.5 * g.dy + j * g.dy, k * g.dz, t)) - a * vector.value(1, g.Nx - 2, j, k)) / (b - c);
            */
            vector.set(2, 0, j, k) = u_boundary.value<2>(0.0, j * g.dy, 0.5 * g.dz + k * g.dz, t);
            Real val2 = vector.value(2, g.Nx - 1, j, k);
            Real gamma_val2 = gamma_field.get(g.Nx - 1, j, k);
            Real sd2 = vector.second_derivative(2, 0, g.Nx - 1, j, k);
            Real rhs_val2 = val2 - gamma_val2 * sd2;
            auto a2 = -gamma_val2 / (g.dx * g.dx);
            auto b2 = 1.0 + 2.0 * gamma_val2 / (g.dx * g.dx);
            auto c2 = -gamma_val2 / (g.dx * g.dx);
            vector.set(2, g.Nx - 1, j, k) = ((rhs_val2 - 2 * c2 * u_boundary.value<2>(Lx, j * g.dy, 0.5 * g.dz + k * g.dz, t)) - a2 * vector.value(2, g.Nx - 2, j, k)) / (b2 - c2);
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

void initialize_fields(const Grid &g,
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

void construct_g_function(VectorVariable &g_function, const Grid &g, BoundaryFunctions &u_boundary,
                          Real t, Real nu, VectorVariable &u_n, VectorVariable &eta_n, VectorVariable &zeta_n)
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

                    // Compute analytical forcing term f = du/dt - nu*Laplacian(u)
                    Real f_val;
                    if (comp == 0)
                    {
                        Real u_t = cos(t) * sin(x) * sin(y) * sin(z);
                        Real laplacian_u = -3.0 * sin(t) * sin(x) * sin(y) * sin(z);
                        f_val = u_t - nu * laplacian_u;
                    }
                    else if (comp == 1)
                    {
                        Real v_t = cos(t) * cos(x) * cos(y) * cos(z);
                        Real laplacian_v = -3.0 * sin(t) * cos(x) * cos(y) * cos(z);
                        f_val = v_t - nu * laplacian_v;
                    }
                    else
                    {
                        Real w_t = cos(t) * cos(x) * sin(y) * (sin(z) + cos(z));
                        Real laplacian_w = -3.0 * sin(t) * cos(x) * sin(y) * (sin(z) + cos(z));
                        f_val = w_t - nu * laplacian_w;
                    }

                    // Compute numerical second derivatives from current solution
                    Real dxx_eta = eta_n.second_derivative(comp, 0, i, j, k);
                    Real dyy_zeta = zeta_n.second_derivative(comp, 1, i, j, k);
                    Real dzz_u = u_n.second_derivative(comp, 2, i, j, k);

                    // g = f - (nu/2)*(dxx + dyy + dzz)
                    Real g_val = f_val - (nu / 2.0) * (dxx_eta + dyy_zeta + dzz_u);

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
           Real &l2_error,
           Real &time,
           bool parallel = false)
{
    auto start_time = std::chrono::high_resolution_clock::now();
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

        // Compute true solution at current time for error computation
        initialize_fields(g, solver.gamma_field, u_true_next, rhs_true_next, solver.u_boundary, t_current, nu);

        // Construct g_function: g = f - (nu/2)*(dxx_eta + dyy_zeta + dzz_u)
        construct_g_function(g_function, g, solver.u_boundary, t_current, nu, u_n, eta_n, zeta_n);
        compute_xi_function(xi_function, g_function, g, u_n, beta);

        // X-direction splitting: solve for eta_{n+1}
        temp_solution.set_all(0.0f);
        rhs = xi_function - eta_n;
        solver.solve<0>(rhs, temp_solution, x_handler, parallel);
        eta_2 = temp_solution + eta_n;

        // Y-direction splitting: solve for zeta_{n+1}
        rhs = eta_2 - zeta_n;
        temp_solution.set_all(0.0f);
        solver.solve<1>(rhs, temp_solution, y_handler, parallel);
        zeta_2 = temp_solution + zeta_n;

        // Z-direction splitting: solve for u_{n+1}
        rhs = zeta_2 - u_n;
        temp_solution.set_all(0.0f);
        solver.solve<2>(rhs, temp_solution, z_handler, parallel);
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
            std::cout << "Error at step " << step + 1 << ": " << compute_L2_error(u_true_next, u_n, g) << std::scientific << std::setprecision(8) << std::endl;
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration<Real>(end_time - start_time).count();

    // Compute L2 error at final time only
    initialize_fields(g, solver.gamma_field, u_true_next, rhs_true_next, solver.u_boundary, t_current - g.dt, nu);
    l2_error = compute_L2_error(u_true_next, u_n, g);
}

// ===============================================================
// 7. Main function
// ===============================================================
int main()
{
    const Real two_pi = 6.0;

    // Open output file
    std::ofstream outfile("test_heat_equation.txt");
    outfile << "# Convergence study for momentum x-direction solver\n";
    outfile << "# Nx Ny Nz dx dt L2_error convergence_rate\n";
    outfile << std::scientific << std::setprecision(8);

    // Test multiple grid resolutions
    std::vector<Dim> grid_sizes = {10, 20};
    std::vector<Real> errors;
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
        // Real dt_test = 0.001 * base_dx;
        Real dt_test = 0.0025;
        Grid g = setup_grid(two_pi, two_pi, two_pi, N, N, N, dt_test);

        printf("\n=================================================\n");
        printf("Grid: Nx=%d, Ny=%d, Nz=%d, dx=%f, dy=%f, dz=%f, dt=%f\n",
               g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz, g.dt);
        printf("=================================================\n");

        ScalarVariable gamma_field(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
        Real nu = 0.1f;
        Real beta = 1.0f;
        gamma_field.set_all(((g.dt * nu) / (Real(2.0) * beta)));
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

        initialize_fields(g, gamma_field, u_1_sequential, rhs_1_sequential, u_boundary, g.dt, nu);
        initialize_fields(g, gamma_field, eta_1_sequential, rhs_1_sequential, u_boundary, g.dt, nu);
        initialize_fields(g, gamma_field, zeta_1_sequential, rhs_1_sequential, u_boundary, g.dt, nu);

        initialize_fields(g, gamma_field, u_1_parallel, rhs_1_parallel, u_boundary, g.dt, nu);
        initialize_fields(g, gamma_field, eta_1_parallel, rhs_1_parallel, u_boundary, g.dt, nu);
        initialize_fields(g, gamma_field, zeta_1_parallel, rhs_1_parallel, u_boundary, g.dt, nu);

        VelocitySolver solver = setup_solver(g, gamma_field, u_boundary, g.dt, g.dt + g.dt);

        DimensionsHandlerVector x_handler(g.Nx, g.Ny, g.Nz, 0, 1, 2, g.dx);

        DimensionsHandlerVector y_handler(g.Nx, g.Ny, g.Nz, 1, 0, 2, g.dy);

        DimensionsHandlerVector z_handler(g.Nx, g.Ny, g.Nz, 2, 0, 1, g.dz);
        Real T_final = Real(1.0);// * (g.dt);

        Real l2_error = 0.0;
        Real time_sequential = 0.0;
        bool parallel = false;

        // solve(solver,
        //       eta_1_sequential,
        //       eta_2_sequential,
        //       zeta_1_sequential,
        //       zeta_2_sequential,
        //       u_1_sequential,
        //       g_function_sequential,
        //       xi_function_sequential,
        //       g,
        //       x_handler,
        //       y_handler,
        //       z_handler,
        //       g.dt + g.dt,
        //       T_final,
        //       nu,
        //       beta,
        //       l2_error, time_sequential, parallel);

        l2_error = 0.0;
        Real time_parallel = 0.0;
        parallel = true;

        solve(solver,
              eta_1_parallel,
              eta_2_parallel,
              zeta_1_parallel,
              zeta_2_parallel,
              u_1_parallel,
              g_function_parallel,
              xi_function_parallel,
              g,
              x_handler,
              y_handler,
              z_handler,
              g.dt + g.dt,
              T_final,
              nu,
              beta,
              l2_error, time_parallel, parallel);

        // Real speed_up = time_sequential / time_parallel;
        // speed_ups.push_back(speed_up);
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
    printf("Grid Size    dx          dt          L2 Error      Conv. Rate    Speed Up\n");
    printf("---------------------------------------------------------------\n");
    for (size_t i = 0; i < errors.size(); ++i)
    {
        Real rate = (i > 0) ? log(errors[i - 1] / errors[i]) / log(dx_values[i - 1] / dx_values[i]) : 0.0;
        printf("%-12d %.6e  %.6e  %.6e  %.4f  %.4f\n",
               grid_sizes[i], dx_values[i], dt_values[i], errors[i], rate); //, speed_ups[i]);
    }
    printf("=================================================\n");
    printf("Results written to: test_heat_equation.txt\n");

    return 0;
}
