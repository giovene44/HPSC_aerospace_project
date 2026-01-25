#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
#include "ScalarVariable.hpp"
#include "VectorVariable.hpp"
#include "Solver.hpp"

// ===============================================================
// 1. Setup Grid Parameters
// ===============================================================
struct Grid
{
    Dim Nx, Ny, Nz;
    Real dx, dy, dz;
    Real dt;
};

// ---------------------------------------------------------------
// Helpers to control what we sweep and how we measure error
// ---------------------------------------------------------------
enum class ErrorRegion
{
    FullDomain,
    InteriorOnly
};

struct RunResult
{
    Real dx;
    Real dt;
    int nsteps;
    Real L2_u; // velocity error
    Real L2_p; // pressure error
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

// A(t) and A'(t)
static inline Real A_of_t(Real t) { return std::sin(t); }
static inline Real dA_dt(Real t) { return std::cos(t); }

// ===============================================================
// EXACT SOLUTIONS
// ===============================================================

// u = sin(t) * sin(x) * sin(y) * sin(z)
// v = sin(t) * cos(x) * cos(y) * cos(z)
// w = sin(t) * cos(x) * sin(y) * (cos(z) + sin(z))
static inline void uvw_true_at(Real x, Real y, Real z, Real t,
                               Real &u, Real &v, Real &w)
{
    const Real A = A_of_t(t);

    // Precompute trig to avoid redundancy
    const Real sx = std::sin(x), cx = std::cos(x);
    const Real sy = std::sin(y), cy = std::cos(y);
    const Real sz = std::sin(z), cz = std::cos(z);

    u = A * sx * sy * sz;
    v = A * cx * cy * cz;
    w = A * cx * sy * (cz + sz);
}

// Time derivatives
static inline void uvw_t_true_at(Real x, Real y, Real z, Real t,
                                 Real &ut, Real &vt, Real &wt)
{
    const Real Ap = dA_dt(t); // cos(t)

    const Real sx = std::sin(x), cx = std::cos(x);
    const Real sy = std::sin(y), cy = std::cos(y);
    const Real sz = std::sin(z), cz = std::cos(z);

    ut = Ap * sx * sy * sz;
    vt = Ap * cx * cy * cz;
    wt = Ap * cx * sy * (cz + sz);
}

// Fill u_true on your staggered grid
static void fill_u_true(VectorVariable &u_true, const Grid &g, Real t)
{
    for (Dim k = 0; k < g.Nz; ++k)
        for (Dim j = 0; j < g.Ny; ++j)
            for (Dim i = 0; i < g.Nx; ++i)
            {
                // comp 0: u at (x+dx/2, y, z)
                {
                    const Real x = (Real(i) + Real(0.5)) * g.dx;
                    const Real y = Real(j) * g.dy;
                    const Real z = Real(k) * g.dz;

                    Real u, v, w;
                    uvw_true_at(x, y, z, t, u, v, w);
                    u_true.set(0, i, j, k) = u;
                }

                // comp 1: v at (x, y+dy/2, z)
                {
                    const Real x = Real(i) * g.dx;
                    const Real y = (Real(j) + Real(0.5)) * g.dy;
                    const Real z = Real(k) * g.dz;

                    Real u, v, w;
                    uvw_true_at(x, y, z, t, u, v, w);
                    u_true.set(1, i, j, k) = v;
                }

                // comp 2: w at (x, y, z+dz/2)
                {
                    const Real x = Real(i) * g.dx;
                    const Real y = Real(j) * g.dy;
                    const Real z = (Real(k) + Real(0.5)) * g.dz;

                    Real u, v, w;
                    uvw_true_at(x, y, z, t, u, v, w);
                    u_true.set(2, i, j, k) = w;
                }
            }
}

// p = sin(t) * cos(x) * cos(y) * cos(z)
static void fill_p_true(ScalarVariable &p_true, const Grid &g, Real t)
{
    const Real A = A_of_t(t);
    for (Dim k = 0; k < g.Nz; ++k)
        for (Dim j = 0; j < g.Ny; ++j)
            for (Dim i = 0; i < g.Nx; ++i)
            {
                const Real x = Real(i) * g.dx;
                const Real y = Real(j) * g.dy;
                const Real z = Real(k) * g.dz;

                p_true.set(i, j, k) = A * std::cos(x) * std::cos(y) * std::cos(z);
            }
}

// ===============================================================
// Compute analytic forcing for MMS Brinkman–Stokes
// PDE: u_t - nu Δu + (nu/k) u + ∇p = f
//
// For this specific exact solution (products of sin/cos),
// Δu = -3u, Δv = -3v, Δw = -3w.
// This simplifies the logic significantly.
// ===============================================================
static void compute_forcing_analytic(VectorVariable &f,
                                     const Grid &g,
                                     Real t,
                                     Real nu,
                                     Real k_perm) // renamed k to k_perm to avoid loop index confusion
{
    f.set_all(Real(0.0));

    const Real A = std::sin(t);
    // const Real Ap = std::cos(t); // Used inside uvw_t_true_at

    // Precompute constant factor for diffusion and damping
    // f = u_t - nu*(-3u) + (nu/k_perm)*u + grad(p)
    // f = u_t + (3*nu + nu/k_perm)*u + grad(p)
    const Real diff_damp_factor = (Real(3.0) * nu) + (nu / k_perm);

    for (Dim kk = 1; kk < g.Nz - 1; ++kk)
        for (Dim jj = 1; jj < g.Ny - 1; ++jj)
            for (Dim ii = 1; ii < g.Nx - 1; ++ii)
            {
                // --------------------------------------------------
                // Component 0: u equation at (x+dx/2, y, z)
                // Need dp/dx
                // --------------------------------------------------
                {
                    const Real x = (Real(ii) + Real(0.5)) * g.dx;
                    const Real y = Real(jj) * g.dy;
                    const Real z = Real(kk) * g.dz;

                    Real u, v, w;
                    uvw_true_at(x, y, z, t, u, v, w);

                    Real ut, vt, wt;
                    uvw_t_true_at(x, y, z, t, ut, vt, wt);

                    // dp/dx of sin(t)cos(x)cos(y)cos(z)
                    // = -sin(t)sin(x)cos(y)cos(z)
                    const Real dp_dx = -A * std::sin(x) * std::cos(y) * std::cos(z);

                    f.set(0, ii, jj, kk) = ut + diff_damp_factor * u + dp_dx;
                }

                // --------------------------------------------------
                // Component 1: v equation at (x, y+dy/2, z)
                // Need dp/dy
                // --------------------------------------------------
                {
                    const Real x = Real(ii) * g.dx;
                    const Real y = (Real(jj) + Real(0.5)) * g.dy;
                    const Real z = Real(kk) * g.dz;

                    Real u, v, w;
                    uvw_true_at(x, y, z, t, u, v, w);

                    Real ut, vt, wt;
                    uvw_t_true_at(x, y, z, t, ut, vt, wt);

                    // dp/dy of sin(t)cos(x)cos(y)cos(z)
                    // = -sin(t)cos(x)sin(y)cos(z)
                    const Real dp_dy = -A * std::cos(x) * std::sin(y) * std::cos(z);

                    f.set(1, ii, jj, kk) = vt + diff_damp_factor * v + dp_dy;
                }

                // --------------------------------------------------
                // Component 2: w equation at (x, y, z+dz/2)
                // Need dp/dz
                // --------------------------------------------------
                {
                    const Real x = Real(ii) * g.dx;
                    const Real y = Real(jj) * g.dy;
                    const Real z = (Real(kk) + Real(0.5)) * g.dz;

                    Real u, v, w;
                    uvw_true_at(x, y, z, t, u, v, w);

                    Real ut, vt, wt;
                    uvw_t_true_at(x, y, z, t, ut, vt, wt);

                    // dp/dz of sin(t)cos(x)cos(y)cos(z)
                    // = -sin(t)cos(x)cos(y)sin(z)
                    const Real dp_dz = -A * std::cos(x) * std::cos(y) * std::sin(z);

                    f.set(2, ii, jj, kk) = wt + diff_damp_factor * w + dp_dz;
                }
            }
}

// ===============================================================
// Build RHS for CN + Brinkman
// ===============================================================
static void build_rhs(VectorVariable &rhs,
                      const VectorVariable &u_n,
                      const ScalarVariable &p_star,
                      const VectorVariable &f_half,
                      const Grid &g,
                      Real nu,
                      Real k)
{
    const Real beta = (nu * g.dt) / (Real(2.0) * k);
    const Real scale = Real(1.0) / (Real(1.0) + beta);
    const Real diff = (nu * g.dt) / Real(2.0);

    for (int c = 0; c < 3; ++c)
        for (Dim kk = 0; kk < g.Nz; ++kk)
            for (Dim jj = 0; jj < g.Ny; ++jj)
                for (Dim ii = 0; ii < g.Nx; ++ii)
                {
                    Real val = u_n.value(c, ii, jj, kk) + g.dt * f_half.value(c, ii, jj, kk) - beta * u_n.value(c, ii, jj, kk);

                    const bool interior =
                        (ii > 0 && ii < g.Nx - 1 &&
                         jj > 0 && jj < g.Ny - 1 &&
                         kk > 0 && kk < g.Nz - 1);

                    if (interior)
                    {
                        // Explicit CN half diffusion: + (nu dt/2) Lap(u^n)
                        const Real u0 = u_n.value(c, ii, jj, kk);

                        const Real uxx =
                            (u_n.value(c, ii + 1, jj, kk) - Real(2.0) * u0 + u_n.value(c, ii - 1, jj, kk)) / (g.dx * g.dx);
                        const Real uyy =
                            (u_n.value(c, ii, jj + 1, kk) - Real(2.0) * u0 + u_n.value(c, ii, jj - 1, kk)) / (g.dy * g.dy);
                        const Real uzz =
                            (u_n.value(c, ii, jj, kk + 1) - Real(2.0) * u0 + u_n.value(c, ii, jj, kk - 1)) / (g.dz * g.dz);

                        val += diff * (uxx + uyy + uzz);

                        // Predictor pressure gradient
                        const Real dp_dx = (p_star.get(ii + 1, jj, kk) - p_star.get(ii, jj, kk)) / g.dx;
                        const Real dp_dy = (p_star.get(ii, jj + 1, kk) - p_star.get(ii, jj, kk)) / g.dy;
                        const Real dp_dz = (p_star.get(ii, jj, kk + 1) - p_star.get(ii, jj, kk)) / g.dz;

                        if (c == 0)
                            val -= g.dt * dp_dx;
                        else if (c == 1)
                            val -= g.dt * dp_dy;
                        else
                            val -= g.dt * dp_dz;
                    }

                    rhs.set(c, ii, jj, kk) = val * scale;
                }
}

static Real compute_L2_error_velocity(const VectorVariable &u_num,
                                      const Grid &g,
                                      Real t,
                                      ErrorRegion region)
{
    // Exact solution at time t
    VectorVariable u_true(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    fill_u_true(u_true, g, t);

    // Index range
    Dim i0 = 0, i1 = g.Nx;
    Dim j0 = 0, j1 = g.Ny;
    Dim k0 = 0, k1 = g.Nz;

    if (region == ErrorRegion::InteriorOnly)
    {
        i0 = 1;
        i1 = g.Nx - 1;
        j0 = 1;
        j1 = g.Ny - 1;
        k0 = 1;
        k1 = g.Nz - 1;
    }

    Real err2 = Real(0);

    for (int comp = 0; comp < 3; ++comp)
        for (Dim k = k0; k < k1; ++k)
            for (Dim j = j0; j < j1; ++j)
                for (Dim i = i0; i < i1; ++i)
                {
                    const Real diff =
                        u_num.value(comp, i, j, k) -
                        u_true.value(comp, i, j, k);
                    err2 += diff * diff;
                }

    return std::sqrt(err2 * g.dx * g.dy * g.dz);
}

static Real compute_L2_error_pressure(const ScalarVariable &p_num,
                                      const Grid &g,
                                      Real t,
                                      ErrorRegion region)
{
    ScalarVariable p_true(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    fill_p_true(p_true, g, t);

    Dim i0 = 0, i1 = g.Nx;
    Dim j0 = 0, j1 = g.Ny;
    Dim k0 = 0, k1 = g.Nz;

    if (region == ErrorRegion::InteriorOnly)
    {
        i0 = 1;
        i1 = g.Nx - 1;
        j0 = 1;
        j1 = g.Ny - 1;
        k0 = 1;
        k1 = g.Nz - 1;
    }

    Real err2 = Real(0.0);

    for (Dim k = k0; k < k1; ++k)
        for (Dim j = j0; j < j1; ++j)
            for (Dim i = i0; i < i1; ++i)
            {
                const Real diff = p_num.get(i, j, k) - p_true.get(i, j, k);
                err2 += diff * diff;
            }

    return std::sqrt(err2 * g.dx * g.dy * g.dz);
}

static void compute_divergence_cell_center(const VectorVariable &u,
                                           ScalarVariable &div,
                                           const Grid &g)
{
    div.set_all(Real(0.0));

    for (Dim k = 1; k < g.Nz - 1; ++k)
        for (Dim j = 1; j < g.Ny - 1; ++j)
            for (Dim i = 1; i < g.Nx - 1; ++i)
            {
                const Real dudx = (u.value(0, i, j, k) - u.value(0, i - 1, j, k)) / g.dx;
                const Real dvdy = (u.value(1, i, j, k) - u.value(1, i, j - 1, k)) / g.dy;
                const Real dwdz = (u.value(2, i, j, k) - u.value(2, i, j, k - 1)) / g.dz;

                div.set(i, j, k) = dudx + dvdy + dwdz;
            }
}

static RunResult run_mms_velocity_case(Dim Nx, Dim Ny, Dim Nz,
                                       Real dt_try,
                                       Real t0, Real Tfinal,
                                       Real nu, Real k,
                                       ErrorRegion region)
{

    const Real pi = Real(3.14159265358979323846);

    // Grid with placeholder dt; then adjust dt to hit Tfinal exactly
    Grid g = setup_grid(2 * pi, 2 * pi, 2 * pi, Nx, Ny, Nz, dt_try);

    int nsteps = int(std::round((Tfinal - t0) / g.dt));
    if (nsteps < 1)
        nsteps = 1;
    g.dt = (Tfinal - t0) / Real(nsteps);

    // BC consistent with NEW manufactured velocity field
    BoundaryFunctions u_boundary;
    std::vector<std::string> bc = {
        "sin(t)*sin(x)*sin(y)*sin(z)",         // u
        "sin(t)*cos(x)*cos(y)*cos(z)",         // v
        "sin(t)*cos(x)*sin(y)*(cos(z)+sin(z))" // w
    };
    u_boundary.set_string_expression(bc);

    // Pressure BCs
    // The exact pressure p = sin(t)cos(x)cos(y)cos(z) satisfies homogeneous Neumann
    // at 0 and 2pi boundaries (derivatives are 0 because of sin terms).
    // So "0" is correct for both total pressure and pressure correction.
    BoundaryFunctions p_boundary;
    std::vector<std::string> pbc = {
        "0", // x-normal derivative
        "0",
        "0"};
    p_boundary.set_string_expression(pbc);

    // Brinkman/CN params
    const Real beta = (nu * g.dt) / (Real(2.0) * k);
    const Real gamma_eff = (nu * g.dt / Real(2.0)) / (Real(1.0) + beta);

    ScalarVariable gamma_field(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    gamma_field.set_all(gamma_eff);

    VelocitySolver solver(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz, g.dt, gamma_field, u_boundary);
    solver.gamma_field = gamma_field;
    solver.u_boundary = u_boundary;

    PressureSolver psolver(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz, g.dt, p_boundary);
    psolver.p_boundary = p_boundary;

    // Fields
    VectorVariable u_n(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    VectorVariable u_tmp(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    VectorVariable u_np1(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    VectorVariable rhs(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    VectorVariable f_half(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);

    ScalarVariable p_half(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz); // ≈ p^{n-1/2}
    ScalarVariable p_star(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz); // ≈ p^{n+1/2,*}

    ScalarVariable div_u(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    ScalarVariable rhs_p(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    ScalarVariable psi(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    ScalarVariable phi(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    ScalarVariable corr_prev(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    ScalarVariable corr_new(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);

    corr_prev.set_all(Real(0.0));

    // Initial condition for velocity
    fill_u_true(u_n, g, t0);

    // Initial condition for pressure
    fill_p_true(p_half, g, t0 - g.dt / Real(2.0));

    // Time loop
    Real t = t0;

    DimensionsHandlerVector x_vector_handler(g.Nx, g.Ny, g.Nz, 0, 1, 2, g.dx);
    DimensionsHandlerVector y_vector_handler(g.Ny, g.Nx, g.Nz, 1, 0, 2, g.dy);
    DimensionsHandlerVector z_vector_handler(g.Nz, g.Nx, g.Ny, 2, 0, 1, g.dz);
    for (int n = 0; n < nsteps; ++n)
    {
        const Real t_np1 = t + g.dt;
        const Real t_half = t + g.dt / Real(2.0);

        // ---- Pressure predictor
        for (Dim kk = 0; kk < g.Nz; ++kk)
            for (Dim jj = 0; jj < g.Ny; ++jj)
                for (Dim ii = 0; ii < g.Nx; ++ii)
                    p_star.set(ii, jj, kk) = p_half.get(ii, jj, kk) + corr_prev.get(ii, jj, kk);

        // ---- Momentum step
        solver.set_t(t_np1);

        compute_forcing_analytic(f_half, g, t_half, nu, k);

        build_rhs(rhs, u_n, p_star, f_half, g, nu, k);

        solver.block_solver<0>(rhs, u_tmp, x_vector_handler, false);
        solver.block_solver<1>(u_tmp, u_np1, y_vector_handler, false);
        solver.block_solver<2>(u_np1, u_tmp, z_vector_handler, false);

        u_n = u_tmp;

        // ---- Pressure correction
        compute_divergence_cell_center(u_n, div_u, g);

        for (Dim kk = 0; kk < g.Nz; ++kk)
            for (Dim jj = 0; jj < g.Ny; ++jj)
                for (Dim ii = 0; ii < g.Nx; ++ii)
                    rhs_p.set(ii, jj, kk) = -div_u.get(ii, jj, kk) / g.dt;

        psolver.set_t(t_np1);

        psolver.solve_x(rhs_p, psi);
        psolver.solve_y(psi, phi);
        psolver.solve_z(phi, corr_new);

        // ---- Pressure update
        for (Dim kk = 0; kk < g.Nz; ++kk)
            for (Dim jj = 0; jj < g.Ny; ++jj)
                for (Dim ii = 0; ii < g.Nx; ++ii)
                    p_half.set(ii, jj, kk) = p_half.get(ii, jj, kk) + corr_new.get(ii, jj, kk);

        corr_prev = corr_new;

        t = t_np1;
    }

    const Real L2_u = compute_L2_error_velocity(u_n, g, Tfinal, region);
    const Real L2_p = compute_L2_error_pressure(p_half, g, Tfinal, region);

    RunResult out;
    out.dx = g.dx;
    out.dt = g.dt;
    out.nsteps = nsteps;
    out.L2_u = L2_u;
    out.L2_p = L2_p;
    return out;
}

// ---------------------------------------------------------------
// Generic sweep runner (prints table + rates)
// ---------------------------------------------------------------
template <class GetAbscissa>
static void run_sweep_and_print(const std::string &title,
                                const std::string &abscissa_name,
                                const std::vector<std::pair<Dim, Real>> &cases, // (N, dt_try)
                                Real t0, Real Tfinal, Real nu, Real k,
                                ErrorRegion region,
                                GetAbscissa getX)
{
    // std::cout << "=================================================\n";
    // std::cout << title << "\n";
    // std::cout << "t0=" << t0 << "  Tfinal=" << Tfinal << "  nu=" << nu << "\n";
    // std::cout << "Domain: [0,2pi]^3\n";
    // std::cout << "=================================================\n\n";

    std::cout << "Grid    dx            dt            nsteps   L2_error_u(Tfinal)   L2_error_p(Tfinal)   rate_u   rate_p\n";
    std::cout << "--------------------------------------------------------------------------------------------------\n";

    std::vector<Real> X;
    std::vector<Real> E_p;
    std::vector<Real> E_u;

    for (size_t idx = 0; idx < cases.size(); ++idx)
    {
        const Dim N = cases[idx].first;
        const Real dt_try = cases[idx].second;

        RunResult r = run_mms_velocity_case(N, N, N, dt_try, t0, Tfinal, nu, k, region);

        const Real x = getX(r);
        X.push_back(x);
        E_p.push_back(r.L2_p);
        E_u.push_back(r.L2_u);

        Real rate_p = 0.0;
        Real rate_u = 0.0;
        if (idx > 0)
        {
            rate_p = std::log(E_p[idx - 1] / E_p[idx]) / std::log(X[idx - 1] / X[idx]);
            rate_u = std::log(E_u[idx - 1] / E_u[idx]) / std::log(X[idx - 1] / X[idx]);
        }
        std::cout
            << std::scientific << std::setprecision(9)
            << std::setw(5) << N << "  "
            << std::setw(12) << r.dx << "  "
            << std::setw(12) << r.dt << "  "
            << std::setw(6) << r.nsteps << "  "
            << std::setw(16) << r.L2_u << "  "
            << std::setw(16) << r.L2_p << "  "
            << std::fixed << std::setprecision(2)
            << std::setw(7) << rate_u << "  "
            << std::setw(7) << rate_p << "\n";
    }

    std::cout << "--------------------------------------------------------------------------------------------------\n";
}

void test_refine_dx(Real nu, Real k, Real t0, Real Tfinal)
{
    std::cout << std::scientific << std::setprecision(12);

    // N doubles => dx halves
    std::vector<Dim> Ns = {5, 10};

    // dt is fixed
    Real dt0 = Real(2e-4);

    std::vector<std::pair<Dim, Real>> cases;
    cases.reserve(Ns.size());
    for (Dim N : Ns)
    {
        cases.push_back({N, dt0});
    }

    run_sweep_and_print(
        "MMS NS TEST: refine dx (dt fixed; fixed Tfinal)",
        "dx",
        cases,
        t0, Tfinal, nu, k,
        ErrorRegion::FullDomain,
        [](const RunResult &r)
        { return r.dx; });
}

void test_refine_dt(Real nu, Real k, Real t0, Real Tfinal)
{
    std::cout << std::scientific << std::setprecision(12);

    // N fixed => dx fixed
    const Dim N = 80;

    const Real dt0 = Real(2.0);
    const Real refinement = Real(0.5);
    const int steps = 8;

    std::vector<std::pair<Dim, Real>> cases;
    cases.reserve(steps);

    Real dt_try = dt0;
    for (int i = 0; i < steps; ++i)
    {
        dt_try *= refinement;
        cases.push_back({N, dt_try});
    }

    run_sweep_and_print(
        "MMS NS TEST: refine dt (dx fixed; fixed Tfinal)",
        "dt",
        cases,
        t0, Tfinal, nu, k,
        ErrorRegion::InteriorOnly,
        [](const RunResult &r)
        { return r.dt; });
}

int main()
{
    const Real nu = Real(0.1);
    const Real k = Real(1.0);
    const Real t0 = Real(0.0);
    const Real Tfinal = Real(0.005);

    test_refine_dx(nu, k, t0, Tfinal);
    // test_refine_dt(nu, k, t0, Tfinal);
    return 0;
}