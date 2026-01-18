#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
#include "ScalarVariable.hpp"
#include "VectorVariable.hpp"
#include "SolverCorrectMPI.hpp"
#include "MPICommunicator.hpp"
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
    Real time;
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
static void fill_u_true(VectorVariable &u_true, const Grid &g, Real t, const MPITopology3D &topo)
{
    Dim k0, k1, j0, j1, i0, i1;
    k0 = topo.local_k0(g.Nz);
    k1 = topo.local_k1(g.Nz);
    j0 = topo.local_j0(g.Ny);
    j1 = topo.local_j1(g.Ny);
    i0 = topo.local_i0(g.Nx);
    i1 = topo.local_i1(g.Nx);
    for (Dim k = k0; k < k1; ++k)
        for (Dim j = j0; j < j1; ++j)
            for (Dim i = i0; i < i1; ++i)
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

// Manufactured pressure: simplest p = sin(x)
static void fill_p_true(ScalarVariable &p_true, const Grid &g, Real t)
{
    for (Dim k = 0; k < g.Nz; ++k)
        for (Dim j = 0; j < g.Ny; ++j)
            for (Dim i = 0; i < g.Nx; ++i)
            {
                const Real x = Real(i) * g.dx;
                const Real y = Real(j) * g.dy;
                const Real z = Real(k) * g.dz;
                p_true.set(i, j, k) = std::sin(x) * std::sin(t);
            }
}
// ===============================================================
// Compute analytic forcing for MMS Brinkman–Stokes (includes +∇p_true)
//
// PDE:
//   u_t - nu Δu + (nu/k) u + ∇p_true = f
//
// Here we manufacture p_true = sin(x) (independent of t)
// so ∂p/∂x = cos(x), ∂p/∂y=0, ∂p/∂z=0.
// ===============================================================
static void compute_forcing_analytic(VectorVariable &f,
                                     const Grid &g,
                                     Real t,
                                     Real nu,
                                     Real k, const MPITopology3D &topo)
{
    f.set_all(Real(0.0));
    Dim k0, k1, j0, j1, i0, i1;
    k0 = topo.local_k0(g.Nz);
    k1 = topo.local_k1(g.Nz);
    j0 = topo.local_j0(g.Ny);
    j1 = topo.local_j1(g.Ny);
    i0 = topo.local_i0(g.Nx);
    i1 = topo.local_i1(g.Nx);
    const Real A = std::sin(t);
    const Real Ap = std::cos(t);

    for (Dim kk = k0; kk < k1; ++kk)
        for (Dim jj = j0; jj < j1; ++jj)
            for (Dim ii = i0; ii < i1; ++ii)
            {
                // --------------------------------------------------
                // Component 0: u at (x+dx/2, y, z)
                // --------------------------------------------------
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

                    const Real u = A * sx2 * sin2y * sz2;
                    const Real ut = Ap * sx2 * sin2y * sz2;

                    const Real lap_u =
                        A * sin2y *
                        (Real(2.0) * cos2x * sz2 - Real(4.0) * sx2 * sz2 + Real(2.0) * sx2 * cos2z);

                    // +∂p/∂x for p=sin(x)
                    // const Real dp_dx = std::sin(t) * std::cos(x);
                    const Real dp_dx = 0;

                    f.set(0, ii, jj, kk) = ut - nu * lap_u + (nu / k) * u + dp_dx;
                }

                // --------------------------------------------------
                // Component 1: v at (x, y+dy/2, z)
                // --------------------------------------------------
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

                    const Real v = -A * sin2x * sy2 * sz2;
                    const Real vt = -Ap * sin2x * sy2 * sz2;

                    const Real lap_v =
                        A * sin2x *
                        (Real(4.0) * sy2 * sz2 - Real(2.0) * cos2y * sz2 - Real(2.0) * sy2 * cos2z);

                    // p=sin(x) => ∂p/∂y = 0
                    f.set(1, ii, jj, kk) = vt - nu * lap_v + (nu / k) * v;
                }

                // --------------------------------------------------
                // Component 2: w = 0 everywhere
                // --------------------------------------------------
                {
                    f.set(2, ii, jj, kk) = Real(0.0);
                }
            }
}

// ===============================================================
// Build RHS for CN + Brinkman (Option B scaling) with Auteri-style
// pressure predictor:
//
// Momentum step uses:  f_half - Grad(p_star)
// where p_star = p^{n-1/2} + ϕ^{n-1/2}.
//
// PDE manufactured as:
//   u_t - nu Δu + (nu/k)u + ∇p_true = f
//
// So if p_star ≈ p_true, then (f_half - ∇p_star) is consistent.
// ===============================================================
static void build_rhs(VectorVariable &rhs,
                      const VectorVariable &u_n,
                      const ScalarVariable &p_star, // predictor pressure at cell centers
                      const VectorVariable &f_half,
                      const Grid &g,
                      Real nu,
                      Real k,
                      const MPITopology3D &topo)
{
    const Real beta = (nu * g.dt) / (Real(2.0) * k);
    const Real scale = Real(1.0) / (Real(1.0) + beta);
    const Real diff = (nu * g.dt) / Real(2.0);
    Dim k0, k1, j0, j1, i0, i1;
    k0 = topo.local_k0(g.Nz);
    k1 = topo.local_k1(g.Nz);
    j0 = topo.local_j0(g.Ny);
    j1 = topo.local_j1(g.Ny);
    i0 = topo.local_i0(g.Nx);
    i1 = topo.local_i1(g.Nx);
    for (int c = 0; c < 3; ++c)
        for (Dim kk = k0; kk < k1; ++kk)
            for (Dim jj = j0; jj < j1; ++jj)
                for (Dim ii = i0; ii < i1; ++ii)
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

                        // Predictor pressure gradient at staggered locations (MAC-consistent one-sided):
                        /*
                          // u-face: dp/dx ≈ (p(i+1)-p(i))/dx
                        // v-face: dp/dy ≈ (p(j+1)-p(j))/dy
                        // w-face: dp/dz ≈ (p(k+1)-p(k))/dz
                        const Real dp_dx = (p_star.get(ii + 1, jj, kk) - p_star.get(ii, jj, kk)) / g.dx;
                        const Real dp_dy = (p_star.get(ii, jj + 1, kk) - p_star.get(ii, jj, kk)) / g.dy;
                        const Real dp_dz = (p_star.get(ii, jj, kk + 1) - p_star.get(ii, jj, kk)) / g.dz;

                        if (c == 0)
                            val -= g.dt * dp_dx;
                        else if (c == 1)
                            val -= g.dt * dp_dy;
                        else
                            val -= g.dt * dp_dz;
                        */
                    }

                    rhs.set(c, ii, jj, kk) = val * scale;
                }
}

static Real allreduce_sum_real(Real local, MPI_Comm comm)
{
#ifdef USE_MPI
    Real global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_FLOAT, MPI_SUM, comm);
    return global;
#else
    return local;
#endif
}

static Real compute_err2_velocity_local(const VectorVariable &u_num,
                                        const Grid &g,
                                        Real t,
                                        ErrorRegion region,
                                        const MPITopology3D &topo)
{
    VectorVariable u_true(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    fill_u_true(u_true, g, t, topo); // usa la tua fill_u_true (replicata)

    Dim i0 = topo.local_i0(g.Nx), i1 = topo.local_i1(g.Nx);
    Dim j0 = topo.local_j0(g.Ny), j1 = topo.local_j1(g.Ny);
    Dim k0 = topo.local_k0(g.Nz), k1 = topo.local_k1(g.Nz);

    if (region == ErrorRegion::InteriorOnly)
    {
        i0 = 1;
        i1 = g.Nx - 1;
        j0 = 1;
        j1 = g.Ny - 1;
        k0 = 1;
        k1 = g.Nz - 1;
    }

    Real err2_local = Real(0);

    for (int comp = 0; comp < 3; ++comp)
        for (Dim k = k0; k < k1; ++k)
            for (Dim j = j0; j < j1; ++j)
                for (Dim i = i0; i < i1; ++i)
                {
                    const Real diff = u_num.value(comp, i, j, k) - u_true.value(comp, i, j, k);
                    err2_local += diff * diff;
                }

    return err2_local;
}

static Real compute_err2_pressure_local(const ScalarVariable &p_num,
                                        const Grid &g,
                                        Real t,
                                        ErrorRegion region,
                                        const MPITopology3D &topo)
{
    ScalarVariable p_true(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    fill_p_true(p_true, g, t);

    Dim i0 = topo.local_i0(g.Nx), i1 = topo.local_i1(g.Nx);
    Dim j0 = topo.local_j0(g.Ny), j1 = topo.local_j1(g.Ny);
    Dim k0 = topo.local_k0(g.Nz), k1 = topo.local_k1(g.Nz);

    if (region == ErrorRegion::InteriorOnly)
    {
        i0 = 1;
        i1 = g.Nx - 1;
        j0 = 1;
        j1 = g.Ny - 1;
        k0 = 1;
        k1 = g.Nz - 1;
    }

    Real err2_local = Real(0);

    for (Dim k = k0; k < k1; ++k)
        for (Dim j = j0; j < j1; ++j)
            for (Dim i = i0; i < i1; ++i)
            {
                const Real diff = p_num.get(i, j, k) - p_true.get(i, j, k);
                err2_local += diff * diff;
            }

    return err2_local;
}

static void compute_divergence_cell_center(const VectorVariable &u,
                                           ScalarVariable &div,
                                           const Grid &g)
{
    div.set_all(Real(0.0));

    // div is defined on pressure grid (cell centers): i=1..Nx-2 etc.
    for (Dim k = 1; k < g.Nz - 1; ++k)
        for (Dim j = 1; j < g.Ny - 1; ++j)
            for (Dim i = 1; i < g.Nx - 1; ++i)
            {
                // u component stored at (i+1/2,j,k) => use u(i) - u(i-1)
                const Real dudx = (u.value(0, i, j, k) - u.value(0, i - 1, j, k)) / g.dx;
                // v at (i,j+1/2,k)
                const Real dvdy = (u.value(1, i, j, k) - u.value(1, i, j - 1, k)) / g.dy;
                // w at (i,j,k+1/2)
                const Real dwdz = (u.value(2, i, j, k) - u.value(2, i, j, k - 1)) / g.dz;

                div.set(i, j, k) = dudx + dvdy + dwdz;
            }
}

static void subtract_mean(ScalarVariable &p, const Grid &g)
{
    Real sum = Real(0.0);
    Dim count = 0;

    for (Dim k = 1; k < g.Nz - 1; ++k)
        for (Dim j = 1; j < g.Ny - 1; ++j)
            for (Dim i = 1; i < g.Nx - 1; ++i)
            {
                sum += p.get(i, j, k);
                ++count;
            }

    const Real mean = sum / Real(count);

    for (Dim k = 1; k < g.Nz - 1; ++k)
        for (Dim j = 1; j < g.Ny - 1; ++j)
            for (Dim i = 1; i < g.Nx - 1; ++i)
                p.set(i, j, k) = p.get(i, j, k) - mean;
}

static RunResult run_mms_velocity_case(Dim Nx, Dim Ny, Dim Nz,
                                       Real dt_try,
                                       Real t0, Real Tfinal,
                                       Real nu, Real k,
                                       ErrorRegion region, const MPITopology3D &topo)
{
    const Real pi = Real(3.14159265358979323846);

    // Grid with placeholder dt; then adjust dt to hit Tfinal exactly
    Grid g = setup_grid(2 * pi, 2 * pi, 2 * pi, Nx, Ny, Nz, dt_try);

    int nsteps = int(std::round((Tfinal - t0) / g.dt));
    if (nsteps < 1)
        nsteps = 1;
    g.dt = (Tfinal - t0) / Real(nsteps);

    // BC consistent with manufactured velocity field
    BoundaryFunctions u_boundary;
    std::vector<std::string> bc = {
        "sin(t)*sin(x)*sin(x)*sin(2*y)*sin(z)*sin(z)",  // u
        "-sin(t)*sin(2*x)*sin(y)*sin(y)*sin(z)*sin(z)", // v
        "0"                                             // w
    };
    u_boundary.set_string_expression(bc);

    // Pressure Neumann BC for p=sin(x): ∂p/∂x = cos(x), ∂p/∂y=0, ∂p/∂z=0
    BoundaryFunctions p_boundary;
    std::vector<std::string> pbc = {
        "0", // x-normal derivative
        "0",
        "0"};
    p_boundary.set_string_expression(pbc);

    // Brinkman/CN params (your Option B scaling)
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

    // Pressure at half-steps and predictor
    ScalarVariable p_half(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz); // ≈ p^{n-1/2}
    ScalarVariable p_star(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz); // ≈ p^{n+1/2,*}

    // Pressure correction variables (Auteri: ψ, φ, ϕ^{n+1/2})
    ScalarVariable div_u(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    ScalarVariable rhs_p(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    ScalarVariable psi(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    ScalarVariable phi(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    ScalarVariable corr_prev(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz); // ≈ ϕ^{n-1/2}
    ScalarVariable corr_new(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);  // ≈ ϕ^{n+1/2}

    corr_prev.set_all(Real(0.0)); // first step: no previous correction

    // Initial condition for velocity
    fill_u_true(u_n, g, t0, topo);

    // Initialize p at half-step: p^{n-1/2} at t0 - dt/2
    // (for p_true = sin(x) independent of t, this is equivalent to p_true(t0))

    // fill_p_true(p_half, g, t0 - g.dt / Real(2.0));

    // Time loop
    Real t = t0;
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int n = 0; n < nsteps; ++n)
    {
        const Real t_np1 = t + g.dt;
        const Real t_half = t + g.dt / Real(2.0);

        // ---- Momentum step (ADI) with predictor pressure
        solver.set_t(t_np1);

        // f includes +∇p_true (manufactured), evaluated at t^{n+1/2}
        compute_forcing_analytic(f_half, g, t_half, nu, k, topo);

        // RHS uses (f_half - ∇p_star) in an Auteri-consistent way
        build_rhs(rhs, u_n, p_star, f_half, g, nu, k, topo);

        solver.solve_x_only(rhs, u_tmp, topo, false);
        solver.solve_y_only(u_tmp, u_np1, topo, false);
        solver.solve_z_only(u_np1, u_tmp, topo, false);

        u_n = u_tmp; // now u_n is u^{n+1}

        t = t_np1;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    const double elapsed_seconds = elapsed.count();

    const Real L2_u_local = compute_err2_velocity_local(u_n, g, Tfinal, region, topo);

    // pressure is stored at half-step; compare to true pressure at same time
    const Real L2_p_local = compute_err2_pressure_local(p_half, g, Tfinal, region, topo);

    Real err2_u_global = allreduce_sum_real(L2_u_local, topo.cart_comm());
    Real L2_u_global = std::sqrt(err2_u_global * g.dx * g.dy * g.dz);

    Real err2_p_global = allreduce_sum_real(L2_p_local, topo.cart_comm());
    Real L2_p_global = std::sqrt(err2_p_global * g.dx * g.dy * g.dz);

    RunResult out;
    out.dx = g.dx;
    out.dt = g.dt;
    out.nsteps = nsteps;
    out.L2_u = L2_u_global;
    out.L2_p = L2_p_global;
    out.time = static_cast<Real>(elapsed_seconds);
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
                                GetAbscissa getX, const MPITopology3D &topo)
{
    if (topo.cart_rank() == 0)
    {
        std::cout << "=================================================\n";
        std::cout << title << "\n";
        std::cout << "t0=" << t0 << "  Tfinal=" << Tfinal << "  nu=" << nu << "\n";
        std::cout << "Domain: [0,2pi]^3\n";
        std::cout << "=================================================\n\n";

        std::cout << "Grid    dx            dt            nsteps   L2_error_u(Tfinal)     rate         time(s)    \n";
        std::cout << "-----------------------------------------------------------------------\n";
    }
    std::vector<Real> X;
    std::vector<Real> E;

    for (size_t idx = 0; idx < cases.size(); ++idx)
    {
        const Dim N = cases[idx].first;
        const Real dt_try = cases[idx].second;

        RunResult r = run_mms_velocity_case(N, N, N, dt_try, t0, Tfinal, nu, k, region, topo);

        const Real x = getX(r); // either r.dx or r.dt
        X.push_back(x);
        E.push_back(r.L2_u); // track velocity error for rate

        Real rate = 0.0;
        if (idx > 0)
            rate = std::log(E[idx - 1] / E[idx]) / std::log(X[idx - 1] / X[idx]);
        if (topo.cart_rank() == 0)
        {

            std::cout << std::setw(5) << N << "  "
                      << std::setw(12) << r.dx << "  "
                      << std::setw(12) << r.dt << "  "
                      << std::setw(6) << r.nsteps << "  "
                      << std::setw(16) << r.L2_u << "  "
                      << std::setw(7) << rate << "   ";
            std::cout << std::setw(10) << r.time << "\n";
        }
    }
    if (topo.cart_rank() == 0)
        std::cout << "-----------------------------------------------------------------------\n";
}

void test_refine_dx(Real nu, Real k, Real t0, Real Tfinal, const MPITopology3D &topo)
{
    std::cout << std::scientific << std::setprecision(12);

    // N doubles => dx halves
    std::vector<Dim> Ns = {10, 20, 40, 80, 160};

    // dt is fixed (only later adjusted slightly inside run_mms_velocity_case to hit Tfinal exactly)
    Real dt0 = Real(8e-5);

    std::vector<std::pair<Dim, Real>> cases;
    cases.reserve(Ns.size());
    for (Dim N : Ns)
    {
        dt0 = dt0; // keep dt fixed
        cases.push_back({N, dt0});
    }

    run_sweep_and_print(
        "[ ---> MPI <--- ]MMS NS TEST: refine dx (dt fixed; fixed Tfinal)",
        "dx",
        cases,
        t0, Tfinal, nu, k,
        ErrorRegion::FullDomain,
        [](const RunResult &r)
        { return r.dx; }, topo);
}

void test_refine_dt(Real nu, Real k, Real t0, Real Tfinal, const MPITopology3D &topo)
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
        { return r.dt; }, topo);
}

void test_strong_scalability(Real nu, Real k, Real t0, Real Tfinal, const MPITopology3D &topo)
{
    std::cout << std::scientific << std::setprecision(12);

    // N doubles => dx halves
    std::vector<Dim> Ns = {160};

    // dt is fixed (only later adjusted slightly inside run_mms_velocity_case to hit Tfinal exactly)
    Real dt0 = Real(8e-5);

    std::vector<std::pair<Dim, Real>> cases;
    cases.reserve(Ns.size());
    for (Dim N : Ns)
    {
        dt0 = dt0; // keep dt fixed
        cases.push_back({N, dt0});
    }

    run_sweep_and_print(
        "[ ---> MPI <--- ]MMS NS TEST: refine dx (dt fixed; fixed Tfinal)",
        "dx",
        cases,
        t0, Tfinal, nu, k,
        ErrorRegion::FullDomain,
        [](const RunResult &r)
        { return r.dx; }, topo);
}

int main(int argc, char **argv)
{
    // number of processes is an input parameter
    int number_of_processes = 1;

    if (argc > 1)
    {
        number_of_processes = std::atoi(argv[1]);
    }
    else
    {
        std::cerr << "Usage: mpirun -n <num_procs> ./MPI_test_navier_stokes_no_pressure <num_procs>\n";
        return 1;
    }
    // Example: choose a 3D grid Px * Py * Pz
    int Px;
    int Py;
    int Pz;
    if (number_of_processes == 1)
    {
        Px = 1;
        Py = 1;
        Pz = 1;
    }
    else if (number_of_processes == 2)
    {
        Px = 2;
        Py = 1;
        Pz = 1;
    }
    else if (number_of_processes == 4)
    {
        Px = 2;
        Py = 2;
        Pz = 1;
    }
    else if (number_of_processes == 8)
    {
        Px = 2;
        Py = 2;
        Pz = 2;
    }
    else
    {
        std::cerr << "Unsupported number of processes. Use 2, 4, or 8.\n";
        return 1;
    }

    MPICommunicator comm;
    comm.init(&argc, &argv); // MPI_Init inside

    int world_rank = comm.get_rank();
    int world_size = comm.get_size();

    if (world_size != Px * Py * Pz)
    {
        if (world_rank == 0)
            std::cerr << "MPI size must be Px*Py*Pz\n";
        comm.finalize();
        return 1;
    }

    // 1) Create 3D Cartesian topology
    MPITopology3D topo(MPI_COMM_WORLD, Pz, Py, Px);

    const Real nu = Real(0.1);
    const Real k = Real(0.10);
    const Real t0 = Real(0.15);
    const Real Tfinal = Real(0.151);

    // test_refine_dx(nu, k, t0, Tfinal, topo);
    //  test_refine_dt(nu, k, t0, Tfinal);
    test_strong_scalability(nu, k, t0, Tfinal, topo);
    comm.finalize(); // MPI_Finalize inside
    return 0;
}
