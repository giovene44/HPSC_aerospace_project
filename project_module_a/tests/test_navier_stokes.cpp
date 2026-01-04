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

// ================= Manufactured NS (div-free, no-slip) =================
// Streamfunction: psi = sin(t) * sin^2(x) * sin^2(y) * sin^2(z)
// Velocity: u = dpsi/dy, v = -dpsi/dx, w = 0  (exactly divergence-free)
//
// This file provides:
//   - uvw_true_at(...)      : manufactured velocity at a physical point
//   - uvw_t_true_at(...)    : time derivative at a physical point
//   - fill_u_true(...)      : fill VectorVariable on your staggered grid
//   - fill_p_true(...)      : p=0 everywhere (ScalarVariable)
//   - compute_forcing_NS(...) : f = u_t + (u·∇)u + ∇p - nu Δu
//
// IMPORTANT:
// - This forcing uses simple centered differences on each component’s own index lattice.
//   That is OK for MMS verification, but it is not the same as a fully consistent
//   staggered-grid convection discretization (MAC) unless you implement the same
//   interpolation/flux form your solver uses.

#include <cmath>

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

static inline Real lap_comp(const VectorVariable &u, int c, Dim i, Dim j, Dim k,
                            Real dx, Real dy, Real dz)
{
    const Real u0 = u.value(c, i, j, k);

    const Real uxx = (u.value(c, i + 1, j, k) - Real(2.0) * u0 + u.value(c, i - 1, j, k)) / (dx * dx);
    const Real uyy = (u.value(c, i, j + 1, k) - Real(2.0) * u0 + u.value(c, i, j - 1, k)) / (dy * dy);
    const Real uzz = (u.value(c, i, j, k + 1) - Real(2.0) * u0 + u.value(c, i, j, k - 1)) / (dz * dz);

    return uxx + uyy + uzz;
}

// Compute f = u_t + (u·∇)u + ∇p - nu Δu on interior points
static void compute_forcing_NS(VectorVariable &f,
                               const VectorVariable &u_true,
                               const ScalarVariable &p_true,
                               const Grid &g,
                               Real t,
                               Real nu)
{
    f.set_all(Real(0.0));

    // Only where centered stencils are valid
    for (Dim k = 1; k < g.Nz - 1; ++k)
        for (Dim j = 1; j < g.Ny - 1; ++j)
            for (Dim i = 1; i < g.Nx - 1; ++i)
            {
                // ---------------- comp 0 ----------------
                {
                    const Real x = (Real(i) + Real(0.5)) * g.dx;
                    const Real y = Real(j) * g.dy;
                    const Real z = Real(k) * g.dz;

                    Real ut, vt, wt;
                    uvw_t_true_at(x, y, z, t, ut, vt, wt);

                    const Real u = u_true.value(0, i, j, k);
                    const Real v = u_true.value(1, i, j, k);
                    const Real w = u_true.value(2, i, j, k);

                    const Real du_dx = ddx_c_comp(u_true, 0, i, j, k, g.dx);
                    const Real du_dy = ddy_c_comp(u_true, 0, i, j, k, g.dy);
                    const Real du_dz = ddz_c_comp(u_true, 0, i, j, k, g.dz);

                    const Real conv = u * du_dx + v * du_dy + w * du_dz;

                    const Real dp_dx = ddx_c(p_true, i, j, k, g.dx);

                    const Real lap = lap_comp(u_true, 0, i, j, k, g.dx, g.dy, g.dz);

                    f.set(0, i, j, k) = ut + conv + dp_dx - nu * lap;
                }

                // ---------------- comp 1 ----------------
                {
                    const Real x = Real(i) * g.dx;
                    const Real y = (Real(j) + Real(0.5)) * g.dy;
                    const Real z = Real(k) * g.dz;

                    Real ut, vt, wt;
                    uvw_t_true_at(x, y, z, t, ut, vt, wt);

                    const Real u = u_true.value(0, i, j, k);
                    const Real v = u_true.value(1, i, j, k);
                    const Real w = u_true.value(2, i, j, k);

                    const Real dv_dx = ddx_c_comp(u_true, 1, i, j, k, g.dx);
                    const Real dv_dy = ddy_c_comp(u_true, 1, i, j, k, g.dy);
                    const Real dv_dz = ddz_c_comp(u_true, 1, i, j, k, g.dz);

                    const Real conv = u * dv_dx + v * dv_dy + w * dv_dz;

                    const Real dp_dy = ddy_c(p_true, i, j, k, g.dy);

                    const Real lap = lap_comp(u_true, 1, i, j, k, g.dx, g.dy, g.dz);

                    f.set(1, i, j, k) = vt + conv + dp_dy - nu * lap;
                }

                // ---------------- comp 2 ----------------
                {
                    const Real x = Real(i) * g.dx;
                    const Real y = Real(j) * g.dy;
                    const Real z = (Real(k) + Real(0.5)) * g.dz;

                    Real ut, vt, wt;
                    uvw_t_true_at(x, y, z, t, ut, vt, wt);

                    const Real u = u_true.value(0, i, j, k);
                    const Real v = u_true.value(1, i, j, k);
                    const Real w = u_true.value(2, i, j, k);

                    const Real dw_dx = ddx_c_comp(u_true, 2, i, j, k, g.dx);
                    const Real dw_dy = ddy_c_comp(u_true, 2, i, j, k, g.dy);
                    const Real dw_dz = ddz_c_comp(u_true, 2, i, j, k, g.dz);

                    const Real conv = u * dw_dx + v * dw_dy + w * dw_dz;

                    const Real dp_dz = ddz_c(p_true, i, j, k, g.dz);

                    const Real lap = lap_comp(u_true, 2, i, j, k, g.dx, g.dy, g.dz);

                    f.set(2, i, j, k) = wt + conv + dp_dz - nu * lap;
                }
            }
}
int main()
{
    std::cout << std::scientific << std::setprecision(12);

    const Real pi = Real(3.14159265358979323846);
    const Real nu = Real(0.1);

    // Grid sizes (doubling gives clean refinement)
    std::vector<Dim> grid_sizes = {5, 10, 20, 40, 80};

    // Fixed time interval
    const Real t0 = Real(0.15);
    const Real Tfinal = Real(0.155);

    // Choose a baseline dt for the coarsest grid
    // and scale dt with dx so time error shrinks together with space.
    Real dt_coarse = Real(2e-4); // dt used when N = grid_sizes[0]
    Real refinement = 1;

    std::vector<Real> errors;
    std::vector<Real> dxs;

    std::cout << "=================================================\n";
    std::cout << "MMS NS TEST: refine dx  (fixed dt and Tfinal)\n";
    std::cout << "t0=" << t0 << "  Tfinal=" << Tfinal << "  nu=" << nu << "\n";
    std::cout << "dt_coarse=" << dt_coarse << " at N=" << grid_sizes[0] << "\n";
    std::cout << "Domain: [0,pi]^3\n";
    std::cout << "=================================================\n\n";

    std::cout << "Grid    dx            dt            nsteps   L2_error(Tfinal)   rate\n";
    std::cout << "-----------------------------------------------------------------------\n";

    // First compute dx0 for the coarse grid (so dt scales consistently)
    Grid g0 = setup_grid(pi, pi, pi, grid_sizes[0], grid_sizes[0], grid_sizes[0], dt_coarse);
    const Real dx0 = g0.dx;

    for (size_t idx = 0; idx < grid_sizes.size(); ++idx)
    {
        const Dim N = grid_sizes[idx];
        dt_coarse = dt_coarse * refinement; // refine dt together with dx
        // Setup grid with placeholder dt; we overwrite g.dt right after
        Grid g = setup_grid(pi, pi, pi, N, N, N, dt_coarse);

        // Choose nsteps so that we land exactly on Tfinal
        int nsteps = int(std::round((Tfinal - t0) / g.dt));
        if (nsteps < 1)
            nsteps = 1;
        g.dt = (Tfinal - t0) / Real(nsteps); // exact final time

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
        VectorVariable u_np1(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
        VectorVariable u_tmp(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
        VectorVariable rhs(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);

        // Manufactured forcing ingredients
        VectorVariable u_true_half(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
        VectorVariable f_half(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
        ScalarVariable p_true(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
        fill_p_true(p_true, g, t0);

        // Initial condition
        fill_u_true(u_n, g, t0);

        // Time loop
        Real t = t0;
        for (int n = 0; n < nsteps; ++n)
        {
            const Real t_np1 = t + g.dt;
            const Real t_half = t + g.dt / Real(2.0);

            solver.set_t(t_np1);

            fill_u_true(u_true_half, g, t_half);
            compute_forcing_NS(f_half, u_true_half, p_true, g, t_half, nu);

            // RHS = u^n + dt f^{n+1/2} + (nu dt/2) Lap(u^n)
            rhs = u_n;
            for (int comp = 0; comp < 3; ++comp)
                for (Dim k = 0; k < g.Nz; ++k)
                    for (Dim j = 0; j < g.Ny; ++j)
                        for (Dim i = 0; i < g.Nx; ++i)
                            rhs.set(comp, i, j, k) =
                                rhs.value(comp, i, j, k) + g.dt * f_half.value(comp, i, j, k);

            add_explicit_laplacian_full(u_n, rhs, g, nu);

            // ADI: X -> Y -> Z
            solver.solve_x_only(rhs, u_tmp, false);
            solver.solve_y_only(u_tmp, u_np1, false);
            solver.solve_z_only(u_np1, u_tmp, false);

            u_n = u_tmp;
            t = t_np1;
        }

        // Error vs truth at Tfinal
        VectorVariable u_true_T(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
        fill_u_true(u_true_T, g, Tfinal);

        Real err2 = 0.0;
        for (int comp = 0; comp < 3; ++comp)
            for (Dim k = 0; k < g.Nz; ++k)
                for (Dim j = 0; j < g.Ny; ++j)
                    for (Dim i = 0; i < g.Nx; ++i)
                    {
                        const Real diff = u_n.value(comp, i, j, k) - u_true_T.value(comp, i, j, k);
                        err2 += diff * diff;
                    }

        const Real L2 = std::sqrt(err2 * g.dx * g.dy * g.dz);

        errors.push_back(L2);
        dxs.push_back(g.dx);

        Real rate = 0.0;
        if (idx > 0)
        {
            rate = std::log(errors[idx - 1] / errors[idx]) /
                   std::log(dxs[idx - 1] / dxs[idx]);
        }

        std::cout << std::setw(5) << N << "  "
                  << std::setw(12) << g.dx << "  "
                  << std::setw(12) << g.dt << "  "
                  << std::setw(6) << nsteps << "  "
                  << std::setw(16) << L2 << "  "
                  << std::setw(7) << rate << "\n";
    }

    std::cout << "-----------------------------------------------------------------------\n";
    return 0;
}
