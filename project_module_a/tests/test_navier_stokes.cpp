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
    Real L2;
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
// Setup solver
// ===============================================================
VelocitySolver setup_solver(const Grid &g, ScalarVariable &gamma_field, BoundaryFunctions &u_boundary, Real dt, Real t)
{
    VelocitySolver solver(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz, dt, gamma_field, u_boundary);
    solver.gamma_field = gamma_field;
    solver.u_boundary = u_boundary;
    solver.set_t(t);
    return solver;
}

// A(t) and A'(t)
static inline Real A_of_t(Real t) { return std::sin(t); }
static inline Real dA_dt(Real t) { return std::cos(t); }

// Manufactured velocity at a physical point (x,y,z,t)
static inline void uvw_true_at(Real x, Real y, Real z, Real t,
                               Real &u, Real &v, Real &w)
{
    const Real A = A_of_t(t);

    const Real sx = std::sin(x), cx = std::cos(x);
    const Real sy = std::sin(y), cy = std::cos(y);
    const Real sz = std::sin(z);

    // sin^2(...)
    const Real sx2 = sx * sx;
    const Real sy2 = sy * sy;
    const Real sz2 = sz * sz;

    // u = d/dy [ A * sin^2(x)*sin^2(y)*sin^2(z) ]
    //   = A * sin^2(x) * (2 sin(y)cos(y)) * sin^2(z)
    u = A * sx2 * (Real(2.0) * sy * cy) * sz2;

    // v = -d/dx [ A * sin^2(x)*sin^2(y)*sin^2(z) ]
    //   = -A * (2 sin(x)cos(x)) * sin^2(y) * sin^2(z)
    v = -A * (Real(2.0) * sx * cx) * sy2 * sz2;

    // w = 0
    w = Real(0.0);
}

// Time derivative u_t, v_t, w_t at a physical point (x,y,z,t)
static inline void uvw_t_true_at(Real x, Real y, Real z, Real t,
                                 Real &ut, Real &vt, Real &wt)
{
    const Real Ap = dA_dt(t);

    const Real sx = std::sin(x), cx = std::cos(x);
    const Real sy = std::sin(y), cy = std::cos(y);
    const Real sz = std::sin(z);

    const Real sx2 = sx * sx;
    const Real sy2 = sy * sy;
    const Real sz2 = sz * sz;

    ut = Ap * sx2 * (Real(2.0) * sy * cy) * sz2;
    vt = -Ap * (Real(2.0) * sx * cx) * sy2 * sz2;
    wt = Real(0.0);
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

// Manufactured pressure: simplest p = 0
static void fill_p_true(ScalarVariable &p_true, const Grid & /*g*/, Real /*t*/)
{
    p_true.set_all(Real(0.0));
}

// ===============================================================
// Compute analytic forcing for MMS Brinkman–Stokes
//
// Manufactured field:
// psi = sin(t) * sin^2(x) * sin^2(y) * sin^2(z)
//
// u =  sin(t) * sin^2(x) * sin(2y) * sin^2(z)
// v = -sin(t) * sin(2x) * sin^2(y) * sin^2(z)
// w = 0
//
// PDE:
// u_t - nu Δu + (nu/k) u = f
//
// This function computes f analytically at staggered locations.
// ===============================================================
static void compute_forcing_analytic(VectorVariable &f,
                                     const Grid &g,
                                     Real t,
                                     Real nu,
                                     Real k)
{
    f.set_all(Real(0.0));

    const Real A = std::sin(t);
    const Real Ap = std::cos(t);

    for (Dim kk = 1; kk < g.Nz - 1; ++kk)
        for (Dim jj = 1; jj < g.Ny - 1; ++jj)
            for (Dim ii = 1; ii < g.Nx - 1; ++ii)
            {
                // ==================================================
                // Component 0: u at (x+dx/2, y, z)
                // ==================================================
                {
                    const Real x = (Real(ii) + Real(0.5)) * g.dx;
                    const Real y = Real(jj) * g.dy;
                    const Real z = Real(kk) * g.dz;

                    const Real sx = std::sin(x), cx = std::cos(x);
                    const Real sy = std::sin(y), cy = std::cos(y);
                    const Real sz = std::sin(z), cz = std::cos(z);

                    const Real sx2 = sx * sx;
                    const Real sz2 = sz * sz;

                    const Real sin2y = Real(2.0) * sy * cy;
                    const Real cos2x = cx * cx - sx * sx;
                    const Real cos2z = cz * cz - sz * sz;

                    // u and u_t
                    const Real u = A * sx2 * sin2y * sz2;
                    const Real ut = Ap * sx2 * sin2y * sz2;

                    // Laplacian of u
                    const Real lap_u =
                        A * sin2y *
                        (Real(2.0) * cos2x * sz2 - Real(4.0) * sx2 * sz2 + Real(2.0) * sx2 * cos2z);

                    f.set(0, ii, jj, kk) = ut - nu * lap_u + (nu / k) * u;
                }

                // ==================================================
                // Component 1: v at (x, y+dy/2, z)
                // ==================================================
                {
                    const Real x = Real(ii) * g.dx;
                    const Real y = (Real(jj) + Real(0.5)) * g.dy;
                    const Real z = Real(kk) * g.dz;

                    const Real sx = std::sin(x), cx = std::cos(x);
                    const Real sy = std::sin(y), cy = std::cos(y);
                    const Real sz = std::sin(z), cz = std::cos(z);

                    const Real sy2 = sy * sy;
                    const Real sz2 = sz * sz;

                    const Real sin2x = Real(2.0) * sx * cx;
                    const Real cos2y = cy * cy - sy * sy;
                    const Real cos2z = cz * cz - sz * sz;

                    // v and v_t
                    const Real v = -A * sin2x * sy2 * sz2;
                    const Real vt = -Ap * sin2x * sy2 * sz2;

                    // Laplacian of v
                    const Real lap_v =
                        A * sin2x *
                        (Real(4.0) * sy2 * sz2 - Real(2.0) * cos2y * sz2 - Real(2.0) * sy2 * cos2z);

                    f.set(1, ii, jj, kk) = vt - nu * lap_v + (nu / k) * v;
                }

                // ==================================================
                // Component 2: w = 0 everywhere
                // ==================================================
                {
                    f.set(2, ii, jj, kk) = Real(0.0);
                }
            }
}

// ===============================================================
// Build RHS for CN + Brinkman (Option B: scaled gamma & RHS)
//
// RHS = ( u^n + dt f^{n+1/2}
//         - beta u^n
//         + (nu dt/2) * Lap_h(u^n) ) / (1 + beta)
//
// beta = nu*dt/(2k)
// ===============================================================
static void build_rhs(VectorVariable &rhs,
                      const VectorVariable &u_n,
                      const VectorVariable &f_half,
                      const Grid &g,
                      Real nu,
                      Real k)
{
    const Real beta = (nu * g.dt) / (Real(2.0) * k);
    const Real scale = Real(1.0) / (Real(1.0) + beta);
    const Real diff = (nu * g.dt) / Real(2.0);

    for (int c = 0; c < 3; ++c)
        for (Dim k0 = 0; k0 < g.Nz; ++k0)
            for (Dim j0 = 0; j0 < g.Ny; ++j0)
                for (Dim i0 = 0; i0 < g.Nx; ++i0)
                {
                    // Base RHS terms
                    Real val =
                        u_n.value(c, i0, j0, k0) + g.dt * f_half.value(c, i0, j0, k0) - beta * u_n.value(c, i0, j0, k0);

                    // Interior: add explicit Laplacian
                    if (i0 > 0 && i0 < g.Nx - 1 &&
                        j0 > 0 && j0 < g.Ny - 1 &&
                        k0 > 0 && k0 < g.Nz - 1)
                    {
                        const Real u0 = u_n.value(c, i0, j0, k0);

                        const Real uxx =
                            (u_n.value(c, i0 + 1, j0, k0) - Real(2.0) * u0 + u_n.value(c, i0 - 1, j0, k0)) / (g.dx * g.dx);

                        const Real uyy =
                            (u_n.value(c, i0, j0 + 1, k0) - Real(2.0) * u0 + u_n.value(c, i0, j0 - 1, k0)) / (g.dy * g.dy);

                        const Real uzz =
                            (u_n.value(c, i0, j0, k0 + 1) - Real(2.0) * u0 + u_n.value(c, i0, j0, k0 - 1)) / (g.dz * g.dz);

                        val += diff * (uxx + uyy + uzz);
                    }

                    // Option B scaling
                    rhs.set(c, i0, j0, k0) = val * scale;
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

// ---------------------------------------------------------------
// One single function that does the actual run and returns the L2 error
// ---------------------------------------------------------------
static RunResult run_mms_velocity_case(Dim Nx, Dim Ny, Dim Nz,
                                       Real dt_try,
                                       Real t0, Real Tfinal,
                                       Real nu, Real k,
                                       ErrorRegion region)
{
    const Real pi = Real(3.14159265358979323846);

    // Grid with placeholder dt; then we adjust dt to hit Tfinal exactly
    Grid g = setup_grid(2 * pi, 2 * pi, 2 * pi, Nx, Ny, Nz, dt_try);

    int nsteps = int(std::round((Tfinal - t0) / g.dt));
    if (nsteps < 1)
        nsteps = 1;
    g.dt = (Tfinal - t0) / Real(nsteps);

    // BC consistent with manufactured field
    BoundaryFunctions u_boundary;
    std::vector<std::string> bc = {
        "sin(t)*sin(x)*sin(x)*sin(2*y)*sin(z)*sin(z)",  // u
        "-sin(t)*sin(2*x)*sin(y)*sin(y)*sin(z)*sin(z)", // v
        "0"                                             // w
    };
    u_boundary.set_string_expression(bc);

    const Real beta = (nu * g.dt) / (Real(2.0) * k); // β = ν dt / (2k)
    const Real gamma_eff = (nu * g.dt / Real(2.0)) / (Real(1.0) + beta);

    ScalarVariable gamma_field(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    gamma_field.set_all(gamma_eff);

    // Solver
    VelocitySolver solver(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz, g.dt, gamma_field, u_boundary);
    solver.gamma_field = gamma_field;
    solver.u_boundary = u_boundary;

    // Fields
    VectorVariable u_n(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    VectorVariable u_tmp(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    VectorVariable u_np1(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    VectorVariable rhs(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    VectorVariable f_half(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    // ScalarVariable p_true(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    // fill_p_true(p_true, g, t0);

    // Initial condition
    fill_u_true(u_n, g, t0);

    // Time loop
    Real t = t0;
    for (int n = 0; n < nsteps; ++n)
    {
        const Real t_np1 = t + g.dt;
        const Real t_half = t + g.dt / Real(2.0);

        solver.set_t(t_np1);

        compute_forcing_analytic(f_half, g, t_half, nu, k);
        // RHS = u^n + dt f^{n+1/2} + (nu dt/2) Lap(u^n)
        build_rhs(rhs, u_n, f_half, g, nu, k);

        // ADI: X -> Y -> Z
        solver.solve_x_only(rhs, u_tmp, false);
        solver.solve_y_only(u_tmp, u_np1, false);
        solver.solve_z_only(u_np1, u_tmp, false);

        u_n = u_tmp;
        t = t_np1;
    }

    const Real L2 = compute_L2_error_velocity(u_n, g, Tfinal, region);

    RunResult out;
    out.dx = g.dx;
    out.dt = g.dt;
    out.nsteps = nsteps;
    out.L2 = L2;
    return out;
}

// ---------------------------------------------------------------
// Generic sweep runner (prints table + rates)
// x-values can be dx or dt depending on what you pass as "abscissa"
// ---------------------------------------------------------------
template <class GetAbscissa>
static void run_sweep_and_print(const std::string &title,
                                const std::string &abscissa_name,
                                const std::vector<std::pair<Dim, Real>> &cases, // (N, dt_try)
                                Real t0, Real Tfinal, Real nu, Real k,
                                ErrorRegion region,
                                GetAbscissa getX)
{
    std::cout << "=================================================\n";
    std::cout << title << "\n";
    std::cout << "t0=" << t0 << "  Tfinal=" << Tfinal << "  nu=" << nu << "\n";
    std::cout << "Domain: [0,2pi]^3\n";
    std::cout << "=================================================\n\n";

    std::cout << "Grid    dx            dt            nsteps   L2_error(Tfinal)   rate\n";
    std::cout << "-----------------------------------------------------------------------\n";

    std::vector<Real> X;
    std::vector<Real> E;

    for (size_t idx = 0; idx < cases.size(); ++idx)
    {
        const Dim N = cases[idx].first;
        const Real dt_try = cases[idx].second;

        RunResult r = run_mms_velocity_case(N, N, N, dt_try, t0, Tfinal, nu, k, region);

        const Real x = getX(r); // either r.dx or r.dt
        X.push_back(x);
        E.push_back(r.L2);

        Real rate = 0.0;
        if (idx > 0)
            rate = std::log(E[idx - 1] / E[idx]) / std::log(X[idx - 1] / X[idx]);

        std::cout << std::setw(5) << N << "  "
                  << std::setw(12) << r.dx << "  "
                  << std::setw(12) << r.dt << "  "
                  << std::setw(6) << r.nsteps << "  "
                  << std::setw(16) << r.L2 << "  "
                  << std::setw(7) << rate << "\n";
    }

    std::cout << "-----------------------------------------------------------------------\n";
}

void test_refine_dx(Real nu, Real k, Real t0, Real Tfinal)
{
    std::cout << std::scientific << std::setprecision(12);

    // N doubles => dx halves
    std::vector<Dim> Ns = {5, 10, 20, 40, 80, 160};

    // dt is fixed (only later adjusted slightly inside run_mms_velocity_case to hit Tfinal exactly)
    const Real dt0 = Real(2e-4);

    std::vector<std::pair<Dim, Real>> cases;
    cases.reserve(Ns.size());
    for (Dim N : Ns)
        cases.push_back({N, dt0});

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
    const int steps = 10;

    std::vector<std::pair<Dim, Real>> cases;
    cases.reserve(steps);

    Real dt_try = dt0;
    for (int i = 0; i < steps; ++i)
    {
        dt_try *= refinement; // first one is dt0*0.5 (like your current behavior)
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
    const Real k = Real(0.10);
    const Real t0 = Real(0.15);
    const Real Tfinal = Real(0.155);

    test_refine_dx(nu, k, t0, Tfinal);
    // test_refine_dt(nu, k, t0, Tfinal);
    return 0;
}
