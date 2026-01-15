#ifndef SOLVER_HPP
#define SOLVER_HPP
#include <string>
#include <cmath>
#include <functional> // For std::function/lambdas
#ifdef _OPENMP
#include <omp.h>
#endif
#include "Variables.hpp"
#include "ScalarVariable.hpp"
#include "VectorVariable.hpp"
#include "DimensionHandler.hpp"
#include "BoundaryFunctions.hpp"

constexpr bool DEBUG_BLOCK = false; // set to true to enable debug prints

class Solver
{
public:
    // Pure virtual destructor makes the class abstract
    virtual ~Solver() = 0;
    Solver(Dim Nx_, Dim Ny_, Dim Nz_, Real dx_, Real dy_, Real dz_, Real dt_)
        : Nx(Nx_), Ny(Ny_), Nz(Nz_),
          dx(dx_), dy(dy_), dz(dz_), dt(dt_) { t = dt_; }; // solve doesn't solve for t=0 called firstly at t=dt

    template <Dim direction>
    void solve(ScalarVariable &rhs, ScalarVariable &solution, const DimensionsHandlerScalar &dim_handler);
    void advance_time()
    {
        t += dt;
    }

    void set_t(Real t)
    {
        this->t = t;
    }

protected:
    Dim Nx; // Grid points in x
    Dim Ny; // Grid points in y
    Dim Nz; // Grid points in z

    Real dx; // Grid spacing in x
    Real dy; // Grid spacing in y
    Real dz; // Grid spacing in z

    Real t;
    Real dt;
    // Make available to derived classes
    void thomas_algorithm(const std::vector<Real> &a, const std::vector<Real> &b, const std::vector<Real> &c, const std::vector<Real> &rhs, std::vector<Real> &x)
    {
        int n = rhs.size();
        std::vector<Real> c_prime(c.size(), 0.0);
        std::vector<Real> rhs_prime(n, 0.0);

        c_prime[0] = c[0] / b[0];
        rhs_prime[0] = rhs[0] / b[0];

        for (int i = 1; i < n; ++i)
        {
            Real m = Real(1.0) / (b[i] - a[i] * c_prime[i - 1]);
            c_prime[i] = c[i] * m;
            rhs_prime[i] = (rhs[i] - a[i] * rhs_prime[i - 1]) * m;
        }

        x[n - 1] = rhs_prime[n - 1];

        for (int i = n - 2; i >= 0; --i)
        {
            x[i] = rhs_prime[i] - c_prime[i] * x[i + 1];
        }
    }
};

// Definition of pure virtual destructor
inline Solver::~Solver() {}

// =============================================================================================
// ====================================Pressure Solver Class====================================
// =============================================================================================

class PressureSolver : public Solver
{
public:
    BoundaryFunctions &p_boundary;

    PressureSolver(Dim Nx_, Dim Ny_, Dim Nz_,
                   Real dx_, Real dy_, Real dz_,
                   Real dt_,
                   BoundaryFunctions &p_boundary_)
        : Solver(Nx_, Ny_, Nz_, dx_, dy_, dz_, dt_), p_boundary(p_boundary_) {}

    BoundaryFunctions &set_p_boundary() { return p_boundary; }

    void solve_x(ScalarVariable &rhs, ScalarVariable &solution)
    {
        const Real alpha = Real(1.0) / (dx * dx); // α = 1/dx^2

        std::vector<Real> a(Nx, -alpha);
        std::vector<Real> b(Nx, Real(1.0) + Real(2.0) * alpha);
        std::vector<Real> c(Nx, -alpha);
        std::vector<Real> d(Nx), x(Nx);

        // Neumann rows via ghost elimination:
        // left:  (1+2α) ψ0 - 2α ψ1 = rhs0 - 2 g/dx
        a[0] = Real(0.0);
        c[0] = -Real(2.0) * alpha;

        // right: -2α ψ_{N-2} + (1+2α) ψ_{N-1} = rhs_{N-1} + 2 g/dx
        a[Nx - 1] = -Real(2.0) * alpha;
        c[Nx - 1] = Real(0.0);

        for (Dim j = 0; j < Ny; ++j)
            for (Dim k = 0; k < Nz; ++k)
            {
                // Copy rhs line
                for (Dim i = 0; i < Nx; ++i)
                    d[i] = rhs.get(i, j, k);

                // Add Neumann BC contributions to rhs (NOT inside matrix)
                // g = ∂ψ/∂x at boundary.
                const Real gL = p_boundary.value<0>(Real(0.0), j * dy, k * dz, t);
                const Real gR = p_boundary.value<0>((Nx - Real(0.5)) * dx, j * dy, k * dz, t);

                d[0] -= Real(2.0) * gL / dx;
                d[Nx - 1] += Real(2.0) * gR / dx;

                thomas_algorithm(a, b, c, d, x);

                for (Dim i = 0; i < Nx; ++i)
                    solution.set(i, j, k) = x[i];
            }
    }

    void solve_y(ScalarVariable &rhs, ScalarVariable &solution)
    {
        const Real alpha = Real(1.0) / (dy * dy);

        std::vector<Real> a(Ny, -alpha);
        std::vector<Real> b(Ny, Real(1.0) + Real(2.0) * alpha);
        std::vector<Real> c(Ny, -alpha);
        std::vector<Real> d(Ny), x(Ny);

        a[0] = Real(0.0);
        c[0] = -Real(2.0) * alpha;

        a[Ny - 1] = -Real(2.0) * alpha;
        c[Ny - 1] = Real(0.0);

        for (Dim i = 0; i < Nx; ++i)
            for (Dim k = 0; k < Nz; ++k)
            {
                for (Dim j = 0; j < Ny; ++j)
                    d[j] = rhs.get(i, j, k);

                const Real gB = p_boundary.value<1>(i * dx, Real(0.0), k * dz, t);
                const Real gT = p_boundary.value<1>(i * dx, (Ny - Real(0.5)) * dy, k * dz, t);

                d[0] -= Real(2.0) * gB / dy;
                d[Ny - 1] += Real(2.0) * gT / dy;

                thomas_algorithm(a, b, c, d, x);

                for (Dim j = 0; j < Ny; ++j)
                    solution.set(i, j, k) = x[j];
            }
    }

    void solve_z(ScalarVariable &rhs, ScalarVariable &solution)
    {
        const Real alpha = Real(1.0) / (dz * dz);

        std::vector<Real> a(Nz, -alpha);
        std::vector<Real> b(Nz, Real(1.0) + Real(2.0) * alpha);
        std::vector<Real> c(Nz, -alpha);
        std::vector<Real> d(Nz), x(Nz);

        a[0] = Real(0.0);
        c[0] = -Real(2.0) * alpha;

        a[Nz - 1] = -Real(2.0) * alpha;
        c[Nz - 1] = Real(0.0);

        for (Dim i = 0; i < Nx; ++i)
            for (Dim j = 0; j < Ny; ++j)
            {
                for (Dim k = 0; k < Nz; ++k)
                    d[k] = rhs.get(i, j, k);

                const Real gF = p_boundary.value<2>(i * dx, j * dy, Real(0.0), t);
                const Real gB = p_boundary.value<2>(i * dx, j * dy, (Nz - Real(0.5)) * dz, t);

                d[0] -= Real(2.0) * gF / dz;
                d[Nz - 1] += Real(2.0) * gB / dz;

                thomas_algorithm(a, b, c, d, x);

                for (Dim k = 0; k < Nz; ++k)
                    solution.set(i, j, k) = x[k];
            }
    }
};

// =============================================================================================
// ====================================Velocity Solver Class====================================
// =============================================================================================

class VelocitySolver : public Solver
{
public:
    ScalarVariable &gamma_field;
    BoundaryFunctions &u_boundary;
    VelocitySolver(Dim Nx_, Dim Ny_, Dim Nz_, Real dx_, Real dy_, Real dz_, Real dt_, ScalarVariable &gam, BoundaryFunctions &u_bnd)
        : Solver(Nx_, Ny_, Nz_, dx_, dy_, dz_, dt_), gamma_field(gam), u_boundary(u_bnd) {}

    void set_gamma(ScalarVariable &g) { gamma_field = g; }
    BoundaryFunctions &set_u_boundary() { return u_boundary; }

    // =============================================================
    // X-DIRECTION ONLY SOLVE (direction splitting validation)
    // Solves: (I - gamma * dxx) u = rhs   along x
    // =============================================================

    void solve_x_only(VectorVariable &rhs,
                      VectorVariable &solution,
                      bool use_omp = false)
    {
        const Dim N = Nx;
        const Real h = dx;
        const Real h2 = h * h;

        const Real Lx = dx * (Nx - Real(0.5));
        auto is_known_face_x = [&](Dim j, Dim k, int comp) -> bool
        {
            // Matches your previous is_known_face<0>(j,k,comp)
            if (comp == 0)
                return (j == 0 || k == 0);
            if (comp == 1)
                return (j == Ny - 1 || k == 0);
            // comp == 2
            return (j == 0 || k == Nz - 1);
        };

        auto fill_known_face_x = [&](Dim j, Dim k, int comp)
        {
            // Fill ONLY this component along the whole x-line at (j,k)
            const Real y_u = Real(j) * dy;
            const Real y_v = (Real(j) + Real(0.5)) * dy;

            const Real z_w = (Real(k) + Real(0.5)) * dz;
            const Real z_u = Real(k) * dz;

            for (Dim i = 0; i < Nx; ++i)
            {
                const Real x_u = (Real(i) + Real(0.5)) * dx; // u location
                const Real x_vw = Real(i) * dx;              // v,w location

                if (comp == 0)
                {
                    // u(i,j,k) at (x+dx/2, y, z)
                    solution.set(0, i, j, k) = u_boundary.value<0>(x_u, y_u, z_u, t);
                }
                else if (comp == 1)
                {
                    // v(i,j,k) at (x, y+dy/2, z)
                    solution.set(1, i, j, k) = u_boundary.value<1>(x_vw, y_v, z_u, t);
                }
                else
                {
                    // w(i,j,k) at (x, y, z+dz/2)
                    solution.set(2, i, j, k) = u_boundary.value<2>(x_vw, y_u, z_w, t);
                }
            }
        };

        auto solve_line_for_component = [&](Dim j, Dim k, int comp)
        {
            std::vector<Real> a(N, 0.0), b(N, 0.0), c(N, 0.0), d(N, 0.0), x(N, 0.0);
            if (is_known_face_x(j, k, comp))
            {
                fill_known_face_x(j, k, comp);
                return;
            }
            // -------------------------
            // Interior coefficients
            // -------------------------
            for (Dim i = 1; i < N - 1; ++i)
            {
                const Real gamma_val = gamma_field.get(i, j, k);
                const Real coeff = gamma_val / h2;

                a[i] = -coeff;
                b[i] = Real(1.0) + Real(2.0) * coeff;
                c[i] = -coeff;

                d[i] = rhs.value(comp, i, j, k);
            }

            // =========================================================
            // LEFT boundary (x=0): lecture BCs
            // Tangential components: Dirichlet at i=0
            // Normal component: u_{1/2} from incompressibility
            // =========================================================
            {
                a[0] = 0.0;
                b[0] = 1.0;
                c[0] = 0.0;

                // Physical coordinates for this (j,k)
                const Real y = j * dy;
                const Real z = k * dz;

                if (comp == 0)
                {
                    // u component is stored at x = dx/2 -> that's u_{1/2}
                    // u_{1/2} = u(0) + (du/dx)|0 * dx/2
                    // (du/dx)|0 = -(dv/dy)|0 - (dw/dz)|0
                    //
                    // Approximate dv/dy and dw/dz using boundary values (second order form in lecture).
                    // We evaluate v at (x=0, y=j±1/2, z=k) and w at (x=0, y=j, z=k±1/2).
                    const Real u0_wall = u_boundary.value<0>(Real(0.0), y, z, t);

                    const Real v_plus = u_boundary.value<1>(Real(0.0), (Real(j) + Real(0.5)) * dy, z, t);
                    const Real v_minus = u_boundary.value<1>(Real(0.0), (Real(j) - Real(0.5)) * dy, z, t);
                    const Real dv_dy = (v_plus - v_minus) / dy;

                    const Real w_plus = u_boundary.value<2>(Real(0.0), y, (Real(k) + Real(0.5)) * dz, t);
                    const Real w_minus = u_boundary.value<2>(Real(0.0), y, (Real(k) - Real(0.5)) * dz, t);
                    const Real dw_dz = (w_plus - w_minus) / dz;

                    const Real dudx0 = -(dv_dy + dw_dz);

                    const Real u_half = u0_wall + dudx0 * (dx * Real(0.5));
                    d[0] = u_half;
                }
                else if (comp == 1)
                {
                    // v at x=0 is on boundary (staggered): v(i=0) is at x=0
                    d[0] = u_boundary.value<1>(Real(0.0), (Real(j) + Real(0.5)) * dy, z, t);
                }
                else // comp == 2
                {
                    // w at x=0 is on boundary: w(i=0) is at x=0
                    d[0] = u_boundary.value<2>(Real(0.0), y, (Real(k) + Real(0.5)) * dz, t);
                }
            }

            // =========================================================
            // RIGHT boundary (x=Lx): lecture BCs
            // Normal component (u): Dirichlet at i=N-1 (u is located at x=Lx)
            // Tangential components (v,w): ghost elimination at i=N-1
            // =========================================================
            if (comp == 0)
            {
                // u at i=N-1 is at x=(N-0.5)dx = Lx, so Dirichlet
                a[N - 1] = 0.0;
                b[N - 1] = 1.0;
                c[N - 1] = 0.0;

                const Real y = j * dy;
                const Real z = k * dz;
                d[N - 1] = u_boundary.value<0>(Lx, y, z, t);
            }
            else
            {
                // Tangential components: use ghost elimination on last row:
                // a u_{N-2} + (b - c) u_{N-1} = f - 2 c u_ex
                //
                // Here for interior stencil c = -coeff, so -2*c*u_ex = +2*coeff*u_ex
                const Real gammaN = gamma_field.get(N - 1, j, k);
                const Real coeff = gammaN / h2;

                // Row coefficients at i=N-1
                a[N - 1] = -coeff;
                b[N - 1] = (Real(1.0) + Real(2.0) * coeff) - (-coeff); // b - c = 1 + 3*coeff
                c[N - 1] = 0.0;

                const Real y = j * dy;
                const Real z = k * dz;

                Real u_ex = 0.0;
                if (comp == 1)
                    u_ex = u_boundary.value<1>(Lx, (Real(j) + Real(0.5)) * dy, z, t);
                else
                    u_ex = u_boundary.value<2>(Lx, y, (Real(k) + Real(0.5)) * dz, t);

                d[N - 1] = rhs.value(comp, N - 1, j, k) + Real(2.0) * coeff * u_ex;
            }

            // Solve TDMA
            thomas_algorithm(a, b, c, d, x);

            // Write back
            for (Dim i = 0; i < N; ++i)
                solution.set(comp, i, j, k) = x[i];
        };

        // Solve for all (j,k) lines and all 3 components
#ifdef _OPENMP
        if (use_omp)
        {
#pragma omp parallel for collapse(2) default(none) shared(solve_line_for_component)
            for (Dim k = 0; k < Nz; ++k)
                for (Dim j = 0; j < Ny; ++j)
                {
                    solve_line_for_component(j, k, 0);
                    solve_line_for_component(j, k, 1);
                    solve_line_for_component(j, k, 2);
                }
            return;
        }
#endif

        for (Dim k = 0; k < Nz; ++k)
            for (Dim j = 0; j < Ny; ++j)
            {
                solve_line_for_component(j, k, 0);
                solve_line_for_component(j, k, 1);
                solve_line_for_component(j, k, 2);
            }
    }

    // =============================================================
    // Y-DIRECTION ONLY SOLVE (direction splitting validation)
    // Solves: (I - gamma * dyy) u = rhs   along y
    // =============================================================
    void solve_y_only(VectorVariable &rhs,
                      VectorVariable &solution,
                      bool use_omp = false)
    {
        const Dim N = Ny;
        const Real h = dy;
        const Real h2 = h * h;

        const Real Ly = dy * (Ny - Real(0.5));

        // ---- Known face logic for direction = 1 (matches your earlier is_known_face<1>) ----
        auto is_known_face_y = [&](Dim i, Dim k, int comp) -> bool
        {
            if (comp == 0)
                return (i == Nx - 1 || k == 0);
            if (comp == 1)
                return (i == 0 || k == 0);
            // comp == 2
            return (i == 0 || k == Nz - 1);
        };

        // Fill ONLY the requested component along the whole y-line at (i,k)
        auto fill_known_face_y = [&](Dim i, Dim k, int comp)
        {
            const Real x_u = (Real(i) + Real(0.5)) * dx; // u location in x
            const Real x_vw = Real(i) * dx;              // v,w location in x

            const Real z_u = Real(k) * dz;
            const Real z_w = (Real(k) + Real(0.5)) * dz;

            for (Dim j = 0; j < Ny; ++j)
            {
                const Real y_u = Real(j) * dy;               // u,w location in y
                const Real y_v = (Real(j) + Real(0.5)) * dy; // v location in y

                if (comp == 0)
                {
                    // u at (x+dx/2, y, z)
                    solution.set(0, i, j, k) = u_boundary.value<0>(x_u, y_u, z_u, t);
                }
                else if (comp == 1)
                {
                    // v at (x, y+dy/2, z)
                    solution.set(1, i, j, k) = u_boundary.value<1>(x_vw, y_v, z_u, t);
                }
                else
                {
                    // w at (x, y, z+dz/2)
                    solution.set(2, i, j, k) = u_boundary.value<2>(x_vw, y_u, z_w, t);
                }
            }
        };

        auto solve_line_for_component = [&](Dim i, Dim k, int comp)
        {
            std::vector<Real> a(N, 0.0), b(N, 0.0), c(N, 0.0), d(N, 0.0), x(N, 0.0);

            // Known face: skip TDMA
            if (is_known_face_y(i, k, comp))
            {
                fill_known_face_y(i, k, comp);
                return;
            }

            // -------------------------
            // Interior coefficients
            // -------------------------
            for (Dim j = 1; j < N - 1; ++j)
            {
                const Real gamma_val = gamma_field.get(i, j, k); // gamma at (i,j,k)
                const Real coeff = gamma_val / h2;

                a[j] = -coeff;
                b[j] = Real(1.0) + Real(2.0) * coeff;
                c[j] = -coeff;

                d[j] = rhs.value(comp, i, j, k);
            }

            // =========================================================
            // LEFT boundary (y=0): lecture BCs
            // Normal comp=1 (v): incompressibility reconstruction
            // Tangentials (u,w): Dirichlet
            // =========================================================
            {
                a[0] = 0.0;
                b[0] = 1.0;
                c[0] = 0.0;

                const Real x_u = (Real(i) + Real(0.5)) * dx;
                const Real x_vw = Real(i) * dx;
                const Real z_u = Real(k) * dz;
                const Real z_w = (Real(k) + Real(0.5)) * dz;

                if (comp == 1)
                {
                    // v is stored at y = dy/2 -> that's v_{1/2}
                    // v_{1/2} = v(0) + (dv/dy)|0 * dy/2
                    // (dv/dy)|0 = -(du/dx)|0 - (dw/dz)|0
                    const Real v0_wall = u_boundary.value<1>(x_vw, Real(0.0), z_u, t);

                    const Real u_plus = u_boundary.value<0>(x_u + Real(0.5) * dx, Real(0.0), z_u, t);
                    const Real u_minus = u_boundary.value<0>(x_u - Real(0.5) * dx, Real(0.0), z_u, t);
                    const Real du_dx = (u_plus - u_minus) / dx;

                    const Real w_plus = u_boundary.value<2>(x_vw, Real(0.0), z_w + Real(0.5) * dz, t);
                    const Real w_minus = u_boundary.value<2>(x_vw, Real(0.0), z_w - Real(0.5) * dz, t);
                    const Real dw_dz = (w_plus - w_minus) / dz;

                    const Real dvdy0 = -(du_dx + dw_dz);

                    const Real v_half = v0_wall + dvdy0 * (dy * Real(0.5));
                    d[0] = v_half;
                }
                else if (comp == 0)
                {
                    // u at y=0 is on boundary
                    d[0] = u_boundary.value<0>(x_u, Real(0.0), z_u, t);
                }
                else // comp == 2
                {
                    // w at y=0 is on boundary
                    d[0] = u_boundary.value<2>(x_vw, Real(0.0), z_w, t);
                }
            }

            // =========================================================
            // RIGHT boundary (y=Ly): lecture BCs
            // Normal comp=1 (v): Dirichlet at j=N-1 (v located at y=Ly)
            // Tangentials (u,w): ghost elimination at j=N-1
            // =========================================================
            if (comp == 1)
            {
                a[N - 1] = 0.0;
                b[N - 1] = 1.0;
                c[N - 1] = 0.0;

                const Real x_vw = Real(i) * dx;
                const Real z_u = Real(k) * dz;
                d[N - 1] = u_boundary.value<1>(x_vw, Ly, z_u, t);
            }
            else
            {
                // ghost elimination on last row:
                // a u_{N-2} + (b - c) u_{N-1} = rhs + 2*coeff*u_ex
                const Real gammaN = gamma_field.get(i, N - 1, k);
                const Real coeff = gammaN / h2;

                a[N - 1] = -coeff;
                b[N - 1] = (Real(1.0) + Real(2.0) * coeff) - (-coeff); // 1 + 3*coeff
                c[N - 1] = 0.0;

                const Real x_u = (Real(i) + Real(0.5)) * dx;
                const Real x_vw = Real(i) * dx;
                const Real z_u = Real(k) * dz;
                const Real z_w = (Real(k) + Real(0.5)) * dz;

                Real u_ex = 0.0;
                if (comp == 0)
                    u_ex = u_boundary.value<0>(x_u, Ly, z_u, t);
                else
                    u_ex = u_boundary.value<2>(x_vw, Ly, z_w, t);

                d[N - 1] = rhs.value(comp, i, N - 1, k) + Real(2.0) * coeff * u_ex;
            }

            // Solve TDMA
            thomas_algorithm(a, b, c, d, x);

            // Write back
            for (Dim j = 0; j < N; ++j)
                solution.set(comp, i, j, k) = x[j];
        };

        // Solve for all (i,k) lines and all 3 components
#ifdef _OPENMP
        if (use_omp)
        {
#pragma omp parallel for collapse(2) default(none) shared(solve_line_for_component)
            for (Dim k = 0; k < Nz; ++k)
                for (Dim i = 0; i < Nx; ++i)
                {
                    solve_line_for_component(i, k, 0);
                    solve_line_for_component(i, k, 1);
                    solve_line_for_component(i, k, 2);
                }
            return;
        }
#endif

        for (Dim k = 0; k < Nz; ++k)
            for (Dim i = 0; i < Nx; ++i)
            {
                solve_line_for_component(i, k, 0);
                solve_line_for_component(i, k, 1);
                solve_line_for_component(i, k, 2);
            }
    }

    // =============================================================
    // Z-DIRECTION ONLY SOLVE (direction splitting validation)
    // Solves: (I - gamma * dzz) u = rhs   along z
    // =============================================================
    void solve_z_only(VectorVariable &rhs,
                      VectorVariable &solution,
                      bool use_omp = false)
    {
        const Dim N = Nz;
        const Real h = dz;
        const Real h2 = h * h;

        const Real Lz = dz * (Nz - Real(0.5));

        // ---- Known face logic for direction = 2 (matches your earlier is_known_face<2>) ----
        // Here index_1 = i, index_2 = j
        auto is_known_face_z = [&](Dim i, Dim j, int comp) -> bool
        {
            if (comp == 0)
                return (i == Nx - 1 || j == 0);
            if (comp == 1)
                return (i == 0 || j == Ny - 1);
            // comp == 2
            return (i == 0 || j == 0);
        };

        // Fill ONLY the requested component along the whole z-line at (i,j)
        auto fill_known_face_z = [&](Dim i, Dim j, int comp)
        {
            const Real x_u = (Real(i) + Real(0.5)) * dx; // u location in x
            const Real x_vw = Real(i) * dx;              // v,w location in x

            const Real y_u = Real(j) * dy;               // u,w location in y
            const Real y_v = (Real(j) + Real(0.5)) * dy; // v location in y

            for (Dim k = 0; k < Nz; ++k)
            {
                const Real z_u = Real(k) * dz;               // u,v location in z
                const Real z_w = (Real(k) + Real(0.5)) * dz; // w location in z

                if (comp == 0)
                {
                    // u at (x+dx/2, y, z)
                    solution.set(0, i, j, k) = u_boundary.value<0>(x_u, y_u, z_u, t);
                }
                else if (comp == 1)
                {
                    // v at (x, y+dy/2, z)
                    solution.set(1, i, j, k) = u_boundary.value<1>(x_vw, y_v, z_u, t);
                }
                else
                {
                    // w at (x, y, z+dz/2)
                    solution.set(2, i, j, k) = u_boundary.value<2>(x_vw, y_u, z_w, t);
                }
            }
        };

        auto solve_line_for_component = [&](Dim i, Dim j, int comp)
        {
            std::vector<Real> a(N, 0.0), b(N, 0.0), c(N, 0.0), d(N, 0.0), x(N, 0.0);

            // Known face: skip TDMA
            if (is_known_face_z(i, j, comp))
            {
                fill_known_face_z(i, j, comp);
                return;
            }

            // -------------------------
            // Interior coefficients
            // -------------------------
            for (Dim k = 1; k < N - 1; ++k)
            {
                const Real gamma_val = gamma_field.get(i, j, k);
                const Real coeff = gamma_val / h2;

                a[k] = -coeff;
                b[k] = Real(1.0) + Real(2.0) * coeff;
                c[k] = -coeff;

                d[k] = rhs.value(comp, i, j, k);
            }

            // =========================================================
            // LEFT boundary (z=0): lecture BCs
            // Normal comp=2 (w): incompressibility reconstruction
            // Tangentials (u,v): Dirichlet
            // =========================================================
            {
                a[0] = 0.0;
                b[0] = 1.0;
                c[0] = 0.0;

                const Real x_u = (Real(i) + Real(0.5)) * dx;
                const Real x_vw = Real(i) * dx;

                const Real y_u = Real(j) * dy;
                const Real y_v = (Real(j) + Real(0.5)) * dy;

                if (comp == 2)
                {
                    // w is stored at z = dz/2 -> that's w_{1/2}
                    // w_{1/2} = w(0) + (dw/dz)|0 * dz/2
                    // (dw/dz)|0 = -(du/dx)|0 - (dv/dy)|0
                    const Real w0_wall = u_boundary.value<2>(x_vw, y_u, Real(0.0), t);

                    const Real u_plus = u_boundary.value<0>(x_u + Real(0.5) * dx, y_u, Real(0.0), t);
                    const Real u_minus = u_boundary.value<0>(x_u - Real(0.5) * dx, y_u, Real(0.0), t);
                    const Real du_dx = (u_plus - u_minus) / dx;

                    const Real v_plus = u_boundary.value<1>(x_vw, y_v + Real(0.5) * dy, Real(0.0), t);
                    const Real v_minus = u_boundary.value<1>(x_vw, y_v - Real(0.5) * dy, Real(0.0), t);
                    const Real dv_dy = (v_plus - v_minus) / dy;

                    const Real dwdz0 = -(du_dx + dv_dy);

                    const Real w_half = w0_wall + dwdz0 * (dz * Real(0.5));
                    d[0] = w_half;
                }
                else if (comp == 0)
                {
                    // u at z=0 is on boundary
                    d[0] = u_boundary.value<0>(x_u, y_u, Real(0.0), t);
                }
                else // comp == 1
                {
                    // v at z=0 is on boundary
                    d[0] = u_boundary.value<1>(x_vw, y_v, Real(0.0), t);
                }
            }

            // =========================================================
            // RIGHT boundary (z=Lz): lecture BCs
            // Normal comp=2 (w): Dirichlet at k=N-1 (w located at z=Lz)
            // Tangentials (u,v): ghost elimination at k=N-1
            // =========================================================
            if (comp == 2)
            {
                a[N - 1] = 0.0;
                b[N - 1] = 1.0;
                c[N - 1] = 0.0;

                const Real x_vw = Real(i) * dx;
                const Real y_u = Real(j) * dy;
                d[N - 1] = u_boundary.value<2>(x_vw, y_u, Lz, t);
            }
            else
            {
                const Real gammaN = gamma_field.get(i, j, N - 1);
                const Real coeff = gammaN / h2;

                a[N - 1] = -coeff;
                b[N - 1] = (Real(1.0) + Real(2.0) * coeff) - (-coeff); // 1 + 3*coeff
                c[N - 1] = 0.0;

                const Real x_u = (Real(i) + Real(0.5)) * dx;
                const Real x_vw = Real(i) * dx;

                const Real y_u = Real(j) * dy;
                const Real y_v = (Real(j) + Real(0.5)) * dy;

                Real u_ex = 0.0;
                if (comp == 0)
                    u_ex = u_boundary.value<0>(x_u, y_u, Lz, t);
                else
                    u_ex = u_boundary.value<1>(x_vw, y_v, Lz, t);

                d[N - 1] = rhs.value(comp, i, j, N - 1) + Real(2.0) * coeff * u_ex;
            }

            // Solve TDMA
            thomas_algorithm(a, b, c, d, x);

            // Write back
            for (Dim k = 0; k < N; ++k)
                solution.set(comp, i, j, k) = x[k];
        };

        // Solve for all (i,j) lines and all 3 components
#ifdef _OPENMP
        if (use_omp)
        {
#pragma omp parallel for collapse(2) default(none) shared(solve_line_for_component)
            for (Dim j = 0; j < Ny; ++j)
                for (Dim i = 0; i < Nx; ++i)
                {
                    solve_line_for_component(i, j, 0);
                    solve_line_for_component(i, j, 1);
                    solve_line_for_component(i, j, 2);
                }
            return;
        }
#endif

        for (Dim j = 0; j < Ny; ++j)
            for (Dim i = 0; i < Nx; ++i)
            {
                solve_line_for_component(i, j, 0);
                solve_line_for_component(i, j, 1);
                solve_line_for_component(i, j, 2);
            }
    }
};
#endif // SOLVER_HPP
