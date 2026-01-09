// ===============================================================
// Minimal exact unit test for solve_x_only with Nx=5, comp=0 only
// Manufactured RHS from a chosen u_true on ONE interior line (j,k)=(1,1)
// ===============================================================
//
// What this test does:
// 1) builds a tiny grid Nx=5, Ny=3, Nz=3 (so j=1,k=1 is NOT a known-face for comp=0)
// 2) sets boundary functions to ZERO for u,v,w (so your left u_{1/2} computation gives 0)
// 3) sets gamma_field = 1 everywhere (so coeff = gamma/dx^2 is clean)
// 4) defines a discrete u_true only along the x-line at (j,k)=(1,1):
//      u_true = [0, 4/7, 5/7, 4/7, 0]
// 5) manufactures rhs = A*u_true using EXACTLY the same tridiagonal operator your solver builds
//    (boundary rows are identity, interior rows use -coeff, 1+2coeff, -coeff)
// 6) calls solver.solve_x_only(rhs, u_num, parallel=false)
// 7) prints rhs and the recovered solution along that line and the max error
//
// Drop this into your file (replace your current main with this main), or create a new test file.
//
// NOTE: This uses your existing classes from "navier_stokes_brinkman.hpp":
//   - ScalarVariable, VectorVariable, BoundaryFunctions, VelocitySolver
//
// ===============================================================

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
// Manufactured RHS builder for ONE x-line at (j,k) for comp=0,
// using the SAME discrete operator as solve_x_only builds.
// ===============================================================
static void manufacture_rhs_from_u_true_line_comp0(
    VectorVariable &rhs,
    const VectorVariable &u_true,
    const ScalarVariable &gamma_field,
    const Grid &g,
    Dim j, Dim k)
{
    const Dim N = g.Nx;
    const Real h2 = g.dx * g.dx;

    // Boundary rows in your solve_x_only for comp=0 are Dirichlet-like rows:
    // b=1, d = boundary_value => system enforces u_i = d
    // To manufacture RHS consistent with that: rhs(0)=u_true(0), rhs(N-1)=u_true(N-1).
    rhs.set(0, 0, j, k) = u_true.value(0, 0, j, k);
    rhs.set(0, N - 1, j, k) = u_true.value(0, N - 1, j, k);

    // Interior rows: rhs_i = a*u_{i-1} + b*u_i + c*u_{i+1}
    // with a=-coeff, b=1+2coeff, c=-coeff and coeff = gamma/h^2
    for (Dim i = 1; i < N - 1; ++i)
    {
        const Real gamma_val = gamma_field.get(i, j, k);
        const Real coeff = gamma_val / h2;

        const Real a = -coeff;
        const Real b = Real(1.0) + Real(2.0) * coeff;
        const Real c = -coeff;

        const Real uim1 = u_true.value(0, i - 1, j, k);
        const Real ui = u_true.value(0, i, j, k);
        const Real uip1 = u_true.value(0, i + 1, j, k);

        rhs.set(0, i, j, k) = a * uim1 + b * ui + c * uip1;
    }
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
} // Add this helper: FULL explicit Laplacian (x+y+z) on interior
static void add_explicit_laplacian_full(
    const VectorVariable &u,
    VectorVariable &rhs,
    const Grid &g,
    Real nu)
{
    for (int c = 0; c < 3; ++c)
        for (Dim k = 1; k < g.Nz - 1; ++k)
            for (Dim j = 1; j < g.Ny - 1; ++j)
                for (Dim i = 1; i < g.Nx - 1; ++i)
                {
                    const Real u0 = u.value(c, i, j, k);

                    const Real uxx =
                        (u.value(c, i + 1, j, k) - Real(2.0) * u0 + u.value(c, i - 1, j, k)) / (g.dx * g.dx);

                    const Real uyy =
                        (u.value(c, i, j + 1, k) - Real(2.0) * u0 + u.value(c, i, j - 1, k)) / (g.dy * g.dy);

                    const Real uzz =
                        (u.value(c, i, j, k + 1) - Real(2.0) * u0 + u.value(c, i, j, k - 1)) / (g.dz * g.dz);

                    const Real lap = uxx + uyy + uzz;

                    // Crank–Nicolson explicit half: + (nu*dt/2) * Lap(u^n)
                    rhs.set(c, i, j, k) += (nu * g.dt / Real(2.0)) * lap;
                }
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

// --------- Centered differences for ScalarVariable p ---------
static inline Real ddx_c(const ScalarVariable &q, Dim i, Dim j, Dim k, Real dx)
{
    return (q.get(i + 1, j, k) - q.get(i - 1, j, k)) / (Real(2.0) * dx);
}
static inline Real ddy_c(const ScalarVariable &q, Dim i, Dim j, Dim k, Real dy)
{
    return (q.get(i, j + 1, k) - q.get(i, j - 1, k)) / (Real(2.0) * dy);
}
static inline Real ddz_c(const ScalarVariable &q, Dim i, Dim j, Dim k, Real dz)
{
    return (q.get(i, j, k + 1) - q.get(i, j, k - 1)) / (Real(2.0) * dz);
}

// --------- Centered differences for VectorVariable components ---------
static inline Real ddx_c_comp(const VectorVariable &u, int c, Dim i, Dim j, Dim k, Real dx)
{
    return (u.value(c, i + 1, j, k) - u.value(c, i - 1, j, k)) / (Real(2.0) * dx);
}
static inline Real ddy_c_comp(const VectorVariable &u, int c, Dim i, Dim j, Dim k, Real dy)
{
    return (u.value(c, i, j + 1, k) - u.value(c, i, j - 1, k)) / (Real(2.0) * dy);
}
static inline Real ddz_c_comp(const VectorVariable &u, int c, Dim i, Dim j, Dim k, Real dz)
{
    return (u.value(c, i, j, k + 1) - u.value(c, i, j, k - 1)) / (Real(2.0) * dz);
}

// ===============================================================
// Analytic Laplacian for the manufactured field used in uvw_true_at
// psi = sin(t) * sin^2(x) * sin^2(y) * sin^2(z)
// u = dpsi/dy = sin(t) * sin^2(x) * sin(2y) * sin^2(z)
// v =-dpsi/dx =-sin(t) * sin(2x) * sin^2(y) * sin^2(z)
// w = 0
//
// Returns Δu, Δv, Δw at a physical point (x,y,z,t).
// ===============================================================
static inline void laplacian_uvw_true_at(Real x, Real y, Real z, Real t,
                                         Real &lap_u, Real &lap_v, Real &lap_w)
{
    const Real A = std::sin(t);

    // Common trig
    const Real sx = std::sin(x), cx = std::cos(x);
    const Real sy = std::sin(y), cy = std::cos(y);
    const Real sz = std::sin(z), cz = std::cos(z);

    const Real sx2 = sx * sx;
    const Real sy2 = sy * sy;
    const Real sz2 = sz * sz;

    const Real sin2x = Real(2.0) * sx * cx; // sin(2x)
    const Real sin2y = Real(2.0) * sy * cy; // sin(2y)
    const Real cos2x = cx * cx - sx * sx;   // cos(2x)
    const Real cos2y = cy * cy - sy * sy;   // cos(2y)
    const Real cos2z = cz * cz - sz * sz;   // cos(2z)

    // --------------------------
    // u(x,y,z,t) = A * sin^2(x) * sin(2y) * sin^2(z)
    // Δu = u_xx + u_yy + u_zz
    //
    // u_xx = 2*A*cos(2x)*sin(2y)*sin^2(z)
    // u_yy = -4*A*sin^2(x)*sin(2y)*sin^2(z)
    // u_zz = 2*A*sin^2(x)*sin(2y)*cos(2z)
    // --------------------------
    lap_u =
        A * sin2y *
        (Real(2.0) * cos2x * sz2 - Real(4.0) * sx2 * sz2 + Real(2.0) * sx2 * cos2z);

    // --------------------------
    // v(x,y,z,t) = -A * sin(2x) * sin^2(y) * sin^2(z)
    //
    // v_xx = 4*A*sin(2x)*sin^2(y)*sin^2(z)
    // v_yy = -2*A*sin(2x)*cos(2y)*sin^2(z)
    // v_zz = -2*A*sin(2x)*sin^2(y)*cos(2z)
    // --------------------------
    lap_v =
        A * sin2x *
        (Real(4.0) * sy2 * sz2 - Real(2.0) * cos2y * sz2 - Real(2.0) * sy2 * cos2z);

    // --------------------------
    // w = 0
    // --------------------------
    lap_w = Real(0.0);
}

// Compute f = u_t - nu * Δu (analytic Laplacian), p = 0
static void compute_forcing_analytic(VectorVariable &f,
                                     const Grid &g,
                                     Real t,
                                     Real nu)
{
    f.set_all(Real(0.0));

    for (Dim k = 1; k < g.Nz - 1; ++k)
        for (Dim j = 1; j < g.Ny - 1; ++j)
            for (Dim i = 1; i < g.Nx - 1; ++i)
            {
                // comp 0: u at (x+dx/2, y, z)
                {
                    const Real x = (Real(i) + Real(0.5)) * g.dx;
                    const Real y = Real(j) * g.dy;
                    const Real z = Real(k) * g.dz;

                    Real ut, vt, wt;
                    uvw_t_true_at(x, y, z, t, ut, vt, wt);

                    Real lap_u, lap_v, lap_w;
                    laplacian_uvw_true_at(x, y, z, t, lap_u, lap_v, lap_w);

                    f.set(0, i, j, k) = ut - nu * lap_u;
                }

                // comp 1: v at (x, y+dy/2, z)
                {
                    const Real x = Real(i) * g.dx;
                    const Real y = (Real(j) + Real(0.5)) * g.dy;
                    const Real z = Real(k) * g.dz;

                    Real ut, vt, wt;
                    uvw_t_true_at(x, y, z, t, ut, vt, wt);

                    Real lap_u, lap_v, lap_w;
                    laplacian_uvw_true_at(x, y, z, t, lap_u, lap_v, lap_w);

                    f.set(1, i, j, k) = vt - nu * lap_v;
                }

                // comp 2: w at (x, y, z+dz/2)
                {
                    const Real x = Real(i) * g.dx;
                    const Real y = Real(j) * g.dy;
                    const Real z = (Real(k) + Real(0.5)) * g.dz;

                    Real ut, vt, wt;
                    uvw_t_true_at(x, y, z, t, ut, vt, wt);

                    Real lap_u, lap_v, lap_w;
                    laplacian_uvw_true_at(x, y, z, t, lap_u, lap_v, lap_w);

                    f.set(2, i, j, k) = wt - nu * lap_w; // = 0
                }
            }
}

static void build_momentum_rhs_cn(VectorVariable &rhs,
                                  const VectorVariable &u_n,
                                  const VectorVariable &f_half,
                                  const Grid &g,
                                  Real nu)
{
    // RHS = u^n + dt f^{n+1/2} + (nu dt/2) * Lap(u^n)   (Laplacian added below)
    for (int comp = 0; comp < 3; ++comp)
        for (Dim k = 0; k < g.Nz; ++k)
            for (Dim j = 0; j < g.Ny; ++j)
                for (Dim i = 0; i < g.Nx; ++i)
                {
                    rhs.set(comp, i, j, k) =
                        u_n.value(comp, i, j, k) + g.dt * f_half.value(comp, i, j, k);
                }

    // Adds (nu dt/2) Lap(u_n) into rhs (assuming your function does that)
    add_explicit_laplacian_full(u_n, rhs, g, nu);
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
                                       Real nu,
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

    // gamma = nu*dt/2
    ScalarVariable gamma_field(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    gamma_field.set_all(nu * g.dt / Real(2.0));

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

    // If you really need pressure for something else, keep it;
    // here p_true is not used in the shown snippet (so you can remove it)
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

        compute_forcing_analytic(f_half, g, t_half, nu);

        // RHS = u^n + dt f^{n+1/2} + (nu dt/2) Lap(u^n)
        build_momentum_rhs_cn(rhs, u_n, f_half, g, nu);

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
                                Real t0, Real Tfinal, Real nu,
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

        RunResult r = run_mms_velocity_case(N, N, N, dt_try, t0, Tfinal, nu, region);

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

void test_refine_dx()
{
    std::cout << std::scientific << std::setprecision(12);

    const Real nu = Real(0.1);
    const Real t0 = Real(0.15);
    const Real Tfinal = Real(0.155);

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
        t0, Tfinal, nu,
        ErrorRegion::FullDomain,
        [](const RunResult &r)
        { return r.dx; });
}

void test_refine_dt()
{
    std::cout << std::scientific << std::setprecision(12);

    const Real nu = Real(0.1);
    const Real t0 = Real(0.015);
    const Real Tfinal = Real(5);

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
        t0, Tfinal, nu,
        ErrorRegion::InteriorOnly,
        [](const RunResult &r)
        { return r.dt; });
}

int main()
{
    // test_refine_dx();
    test_refine_dt();
    return 0;
}
