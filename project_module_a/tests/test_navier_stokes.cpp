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
    (void)u_boundary;

    for (int comp = 0; comp < 3; ++comp)
        for (Dim k = 0; k < g.Nz; ++k)
            for (Dim j = 0; j < g.Ny; ++j)
                for (Dim i = 0; i < g.Nx; ++i)
                {
                    Real x, y, z;

                    // staggered locations
                    if (comp == 0)
                    {
                        x = (Real(i) + 0.5) * g.dx;
                        y = Real(j) * g.dy;
                        z = Real(k) * g.dz;
                    }
                    if (comp == 1)
                    {
                        x = Real(i) * g.dx;
                        y = (Real(j) + 0.5) * g.dy;
                        z = Real(k) * g.dz;
                    }
                    if (comp == 2)
                    {
                        x = Real(i) * g.dx;
                        y = Real(j) * g.dy;
                        z = (Real(k) + 0.5) * g.dz;
                    }

                    const Real st = std::sin(t);

                    if (comp == 0)
                        vector.set(0, i, j, k) = st * std::sin(x) * std::cos(y) * std::sin(z);
                    else if (comp == 1)
                        vector.set(1, i, j, k) = -st * std::cos(x) * std::sin(y) * std::sin(z);
                    else
                        vector.set(2, i, j, k) = 0.0;
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

inline bool is_error_dof_x(Dim i, Dim j, Dim k, int comp,
                           Dim Nx, Dim Ny, Dim Nz)
{
    (void)j;
    (void)k;
    (void)Ny;
    (void)Nz;

    // x = 0  (i=0): u reconstructed, v/w Dirichlet
    if (i == 0)
        return (comp != 0);

    // x = Lx (i=Nx-1): u Dirichlet, v/w ghost-eliminated
    if (i == Nx - 1)
        return (comp == 0);

    // interior (and any “known faces” you set exactly) are fine
    return true;
}

inline bool is_error_dof_y(Dim i, Dim j, Dim k, int comp,
                           Dim Nx, Dim Ny, Dim Nz)
{
    (void)i;
    (void)k;
    (void)Nx;
    (void)Nz;

    // y = 0  (j=0): v reconstructed, u/w Dirichlet
    if (j == 0)
        return (comp != 1);

    // y = Ly (j=Ny-1): v Dirichlet, u/w ghost-eliminated
    if (j == Ny - 1)
        return (comp == 1);

    return true;
}

inline bool is_error_dof_z(Dim i, Dim j, Dim k, int comp,
                           Dim Nx, Dim Ny, Dim Nz)
{
    (void)i;
    (void)j;
    (void)Nx;
    (void)Ny;

    // z = 0  (k=0): w reconstructed, u/v Dirichlet
    if (k == 0)
        return (comp != 2);

    // z = Lz (k=Nz-1): w Dirichlet, u/v ghost-eliminated
    if (k == Nz - 1)
        return (comp == 2);

    return true;
}

template <typename MaskFn>
Real compute_L2_error_masked(const VectorVariable &exact,
                             const VectorVariable &num,
                             const Grid &g,
                             MaskFn keep_dof)
{
    Real err2 = 0.0;

    for (int comp = 0; comp < 3; ++comp)
        for (Dim k = 0; k < g.Nz; ++k)
            for (Dim j = 0; j < g.Ny; ++j)
                for (Dim i = 0; i < g.Nx; ++i)
                {
                    if (!keep_dof(i, j, k, comp, g.Nx, g.Ny, g.Nz))
                        continue;

                    const Real d = exact.value(comp, i, j, k) - num.value(comp, i, j, k);
                    err2 += d * d;
                }

    return std::sqrt(err2 * g.dx * g.dy * g.dz);
}

Real compute_L2_error_x(const VectorVariable &exact,
                        const VectorVariable &num,
                        const Grid &g)
{
    return compute_L2_error_masked(exact, num, g, is_error_dof_x);
}

Real compute_L2_error_y(const VectorVariable &exact,
                        const VectorVariable &num,
                        const Grid &g)
{
    return compute_L2_error_masked(exact, num, g, is_error_dof_y);
}

Real compute_L2_error_z(const VectorVariable &exact,
                        const VectorVariable &num,
                        const Grid &g)
{
    return compute_L2_error_masked(exact, num, g, is_error_dof_z);
}

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

// Helper: physical location of component comp at (i,j,k)
static inline void phys_coord(int comp, Dim i, Dim j, Dim k,
                              const Grid &g, Real &x, Real &y, Real &z)
{
    if (comp == 0)
    {
        x = (Real(i) + 0.5) * g.dx;
        y = Real(j) * g.dy;
        z = Real(k) * g.dz;
    }
    if (comp == 1)
    {
        x = Real(i) * g.dx;
        y = (Real(j) + 0.5) * g.dy;
        z = Real(k) * g.dz;
    }
    if (comp == 2)
    {
        x = Real(i) * g.dx;
        y = Real(j) * g.dy;
        z = (Real(k) + 0.5) * g.dz;
    }
}

// Helper: get u(c,i,j,k) but if out of range use BC evaluated at that location
static inline Real get_u_or_bc(const VectorVariable &u,
                               const BoundaryFunctions &ub,
                               const Grid &g,
                               int comp, Dim i, Dim j, Dim k,
                               Real t)
{
    // inside: use stored value
    if (i >= 0 && i < g.Nx && j >= 0 && j < g.Ny && k >= 0 && k < g.Nz)
        return u.value(comp, i, j, k);

    // outside: evaluate the analytic BC at the GHOST LOCATION (no clamping!)
    // Note: i,j,k may be -1 or Nx, etc. That’s OK: your analytic BC is defined for any real x,y,z.
    Real x, y, z;

    if (comp == 0)
    {
        x = (Real(i) + 0.5) * g.dx;
        y = Real(j) * g.dy;
        z = Real(k) * g.dz;
        return ub.value<0>(x, y, z, t);
    }
    if (comp == 1)
    {
        x = Real(i) * g.dx;
        y = (Real(j) + 0.5) * g.dy;
        z = Real(k) * g.dz;
        return ub.value<1>(x, y, z, t);
    }
    // comp == 2
    x = Real(i) * g.dx;
    y = Real(j) * g.dy;
    z = (Real(k) + 0.5) * g.dz;
    return ub.value<2>(x, y, z, t);
}

void add_explicit_laplacian_all_nodes(
    const VectorVariable &u,
    VectorVariable &rhs,
    const Grid &g,
    Real nu,
    const BoundaryFunctions &u_boundary,
    Real t_for_bc)
{
    const Real idx2 = 1.0 / (g.dx * g.dx);
    const Real idy2 = 1.0 / (g.dy * g.dy);
    const Real idz2 = 1.0 / (g.dz * g.dz);

    for (int c = 0; c < 3; ++c)
        for (Dim k = 0; k < g.Nz; ++k)
            for (Dim j = 0; j < g.Ny; ++j)
                for (Dim i = 0; i < g.Nx; ++i)
                {
                    const Real uC = get_u_or_bc(u, u_boundary, g, c, i, j, k, t_for_bc);
                    const Real uIp = get_u_or_bc(u, u_boundary, g, c, i + 1, j, k, t_for_bc);
                    const Real uIm = get_u_or_bc(u, u_boundary, g, c, i - 1, j, k, t_for_bc);
                    const Real uJp = get_u_or_bc(u, u_boundary, g, c, i, j + 1, k, t_for_bc);
                    const Real uJm = get_u_or_bc(u, u_boundary, g, c, i, j - 1, k, t_for_bc);
                    const Real uKp = get_u_or_bc(u, u_boundary, g, c, i, j, k + 1, t_for_bc);
                    const Real uKm = get_u_or_bc(u, u_boundary, g, c, i, j, k - 1, t_for_bc);

                    const Real lap = (uIp - 2 * uC + uIm) * idx2 + (uJp - 2 * uC + uJm) * idy2 + (uKp - 2 * uC + uKm) * idz2;

                    rhs.set(c, i, j, k) += (nu * g.dt * 0.5) * lap;
                }
}

void construct_f_function(VectorVariable &f,
                          ScalarVariable &gamma_field,
                          const Grid &g,
                          Real t,
                          Real nu)
{
    (void)gamma_field;

    for (int comp = 0; comp < 3; ++comp)
        for (Dim k = 0; k < g.Nz; ++k)
            for (Dim j = 0; j < g.Ny; ++j)
                for (Dim i = 0; i < g.Nx; ++i)
                {
                    Real x, y, z;

                    if (comp == 0)
                    {
                        x = (Real(i) + 0.5) * g.dx;
                        y = Real(j) * g.dy;
                        z = Real(k) * g.dz;
                    }
                    if (comp == 1)
                    {
                        x = Real(i) * g.dx;
                        y = (Real(j) + 0.5) * g.dy;
                        z = Real(k) * g.dz;
                    }
                    if (comp == 2)
                    {
                        x = Real(i) * g.dx;
                        y = Real(j) * g.dy;
                        z = (Real(k) + 0.5) * g.dz;
                    }

                    const Real st = std::sin(t);
                    const Real ct = std::cos(t);

                    if (comp == 0)
                    {
                        const Real u = st * std::sin(x) * std::cos(y) * std::sin(z);
                        const Real ut = ct * std::sin(x) * std::cos(y) * std::sin(z);
                        f.set(0, i, j, k) = ut + 3.0 * nu * u;
                    }
                    else if (comp == 1)
                    {
                        const Real v = -st * std::cos(x) * std::sin(y) * std::sin(z);
                        const Real vt = -ct * std::cos(x) * std::sin(y) * std::sin(z);
                        f.set(1, i, j, k) = vt + 3.0 * nu * v;
                    }
                    else
                    {
                        f.set(2, i, j, k) = 0.0; // w=0 => f=0
                    }
                }
}

void solve_direct_test(
    VelocitySolver &solver,
    VectorVariable &u_n,
    VectorVariable &u_np1,
    VectorVariable &f_half,
    const Grid &g,
    const DimensionsHandlerVector &x_handler,
    const DimensionsHandlerVector &y_handler,
    const DimensionsHandlerVector &z_handler,
    BoundaryFunctions &u_boundary,
    Real t_start,
    Real T_final,
    Real nu,
    Real &l2_error,
    bool parallel)
{
    (void)x_handler;
    (void)y_handler;
    (void)z_handler;

    VectorVariable xi(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);        // ξ^{n+1}
    VectorVariable rhs_delta(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz); // changing per stage
    VectorVariable du(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);        // δ (solution of each stage)

    VectorVariable eta(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);  // η state
    VectorVariable zeta(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz); // ζ state

    VectorVariable u_true(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    VectorVariable dummy(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);

    Real t = t_start;
    int nsteps = int((T_final - t_start) / g.dt);

    for (int n = 0; n < nsteps; ++n)
    {
        const Real t_np1 = t + g.dt;
        const Real t_half = t + g.dt / 2.0;

        // -------------------------------------------------
        // g^{n+1/2}  (your manufactured f_half)
        // -------------------------------------------------
        construct_f_function(f_half, solver.gamma_field, g, t_half, nu);

        // -------------------------------------------------
        // ξ^{n+1} = u^n + dt*g^{n+1/2} + (nu dt/2)*Δ u^n
        // (this is what you were calling "rhs")
        // -------------------------------------------------
        xi = u_n;

        for (int comp = 0; comp < 3; ++comp)
            for (Dim k = 0; k < g.Nz; ++k)
                for (Dim j = 0; j < g.Ny; ++j)
                    for (Dim i = 0; i < g.Nx; ++i)
                        xi.set(comp, i, j, k) =
                            xi.value(comp, i, j, k) + g.dt * f_half.value(comp, i, j, k);

        add_explicit_laplacian_all_nodes(u_n, xi, g, nu, u_boundary, t_half);

        // Stage time for BCs/forcing (matches your manufactured forcing)
        solver.set_t(t_np1);

        // Initialize intermediate states at time n
        eta = u_n;
        zeta = u_n;

        // =========================================================
        // X delta stage:
        // (I - γ Dxx) (η^{n+1} - η^n) = ξ^{n+1} - η^n
        // with η^n initialized to u^n
        // =========================================================
        for (int comp = 0; comp < 3; ++comp)
            for (Dim k = 0; k < g.Nz; ++k)
                for (Dim j = 0; j < g.Ny; ++j)
                    for (Dim i = 0; i < g.Nx; ++i)
                        rhs_delta.set(comp, i, j, k) =
                            xi.value(comp, i, j, k) - eta.value(comp, i, j, k);

        solver.solve_x_only_delta(rhs_delta, du, t, t_np1, parallel);

        for (int comp = 0; comp < 3; ++comp)
            for (Dim k = 0; k < g.Nz; ++k)
                for (Dim j = 0; j < g.Ny; ++j)
                    for (Dim i = 0; i < g.Nx; ++i)
                        eta.set(comp, i, j, k) =
                            eta.value(comp, i, j, k) + du.value(comp, i, j, k);

        // =========================================================
        // Y delta stage:
        // (I - γ Dyy) (ζ^{n+1} - ζ^n) = η^{n+1} - ζ^n
        // with ζ^n initialized to u^n
        // =========================================================
        for (int comp = 0; comp < 3; ++comp)
            for (Dim k = 0; k < g.Nz; ++k)
                for (Dim j = 0; j < g.Ny; ++j)
                    for (Dim i = 0; i < g.Nx; ++i)
                        rhs_delta.set(comp, i, j, k) =
                            eta.value(comp, i, j, k) - zeta.value(comp, i, j, k);

        solver.solve_y_only_delta(rhs_delta, du, t, t_np1, parallel);

        for (int comp = 0; comp < 3; ++comp)
            for (Dim k = 0; k < g.Nz; ++k)
                for (Dim j = 0; j < g.Ny; ++j)
                    for (Dim i = 0; i < g.Nx; ++i)
                        zeta.set(comp, i, j, k) =
                            zeta.value(comp, i, j, k) + du.value(comp, i, j, k);

        // =========================================================
        // Z delta stage:
        // (I - γ Dzz) (u^{n+1} - u^n) = ζ^{n+1} - u^n
        // =========================================================
        for (int comp = 0; comp < 3; ++comp)
            for (Dim k = 0; k < g.Nz; ++k)
                for (Dim j = 0; j < g.Ny; ++j)
                    for (Dim i = 0; i < g.Nx; ++i)
                        rhs_delta.set(comp, i, j, k) =
                            zeta.value(comp, i, j, k) - u_n.value(comp, i, j, k);

        solver.solve_z_only_delta(rhs_delta, du, t, t_np1, parallel);

        for (int comp = 0; comp < 3; ++comp)
            for (Dim k = 0; k < g.Nz; ++k)
                for (Dim j = 0; j < g.Ny; ++j)
                    for (Dim i = 0; i < g.Nx; ++i)
                        u_np1.set(comp, i, j, k) =
                            u_n.value(comp, i, j, k) + du.value(comp, i, j, k);

        u_n = u_np1;
        t = t_np1;
    }

    // Error at final time
    initialize_fields(g, solver.gamma_field, u_true, dummy, solver.u_boundary, t, nu);
    l2_error = compute_L2_error(u_true, u_n, g);
}

int main()
{
    const Real two_pi = 6.283185307179586;

    std::ofstream outfile("test_navier_stokes_z_direction.txt");
    outfile << "# Convergence study for x-direction heat solver (DIRECT)\n";
    outfile << "# Nx Ny Nz dx dt L2_error convergence_rate\n";
    outfile << std::scientific << std::setprecision(8);

    std::vector<Dim> grid_sizes = {20, 40};
    std::vector<Real> errors;
    std::vector<Real> dx_values;
    std::vector<Real> dt_values;

    for (Dim N : grid_sizes)
    {
        // -------------------------------------------------
        // Grid and timestep (parabolic scaling)
        // -------------------------------------------------
        Real dx = two_pi / (N - 0.5);
        Real dt = 0.01 * dx; // safe implicit timestep

        Grid g = setup_grid(two_pi, two_pi, two_pi, N, N, N, dt);

        printf("\n=================================================\n");
        printf("Grid: N=%d  dx=%e  dt=%e\n", N, g.dx, g.dt);
        printf("=================================================\n");

        // -------------------------------------------------
        // Physical parameters
        // -------------------------------------------------
        Real nu = 0.1;

        // -------------------------------------------------
        // Gamma field: gamma = nu * dt
        // (VelocitySolver internally divides by h^2)
        // -------------------------------------------------
        ScalarVariable gamma_field(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
        gamma_field.set_all(nu * dt / 2.0);

        // -------------------------------------------------
        // Solution variables
        // -------------------------------------------------
        VectorVariable u_n(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
        VectorVariable u_np1(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
        VectorVariable f_fun(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);

        // -------------------------------------------------
        // Boundary conditions (manufactured solution)
        // -------------------------------------------------
        BoundaryFunctions u_boundary;
        std::vector<std::string> bc = {
            "sin(t)*sin(x)*cos(y)*sin(z)",
            "-sin(t)*cos(x)*sin(y)*sin(z)",
            "0"};
        u_boundary.set_string_expression(bc);

        // -------------------------------------------------
        // Initial condition at t = 0
        // -------------------------------------------------
        VectorVariable rhs_dummy(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
        initialize_fields(g, gamma_field, u_n, rhs_dummy, u_boundary, 0.0, nu);

        // -------------------------------------------------
        // Solver setup
        // -------------------------------------------------
        VelocitySolver solver =
            setup_solver(g, gamma_field, u_boundary, g.dt, 0.0);

        DimensionsHandlerVector x_handler(
            g.Nx, g.Ny, g.Nz, 0, 1, 2, g.dx);

        DimensionsHandlerVector y_handler(
            g.Nx, g.Ny, g.Nz, 1, 0, 2, g.dy);

        DimensionsHandlerVector z_handler(
            g.Nx, g.Ny, g.Nz, 2, 0, 1, g.dz);

        // -------------------------------------------------
        // Time integration
        // -------------------------------------------------
        Real T_final = 0.1;
        Real l2_error = 0.0;

        solve_direct_test(
            solver,
            u_n,
            u_np1,
            f_fun,
            g,
            x_handler,
            y_handler,
            z_handler,
            u_boundary,
            0.0,
            T_final,
            nu,
            l2_error,
            true);

        // -------------------------------------------------
        // Store results
        // -------------------------------------------------
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
    printf("CONVERGENCE STUDY SUMMARY (DIRECT SOLVE)\n");
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
    printf("Results written to test_navier_stokes_z_direction.txt\n");

    return 0;
}
