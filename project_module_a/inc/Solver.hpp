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
    template <Dim direction>
    void apply_bc(ScalarVariable &rhs)
    {
        if constexpr (direction == 0) // X direction
        {
            for (Dim index_1 = 0; index_1 < Ny; ++index_1)
            {
                for (Dim index_2 = 0; index_2 < Nz; ++index_2)
                {
                    rhs.set(0, index_1, index_2) = rhs.get(0, index_1, index_2) /*+ Real(2.0) * dx * (-Real(1.0) / (dx * dx)) * p_boundary.first_derivative<0>(0, index_1 * dy, index_2 * dz, t)*/; // Probably we should use the exact value
                    rhs.set(Nx - 1, index_1, index_2) = rhs.get(Nx - 1, index_1, index_2) /*- dx * (-Real(1.0) / (dx * dx)) * p_boundary.first_derivative<0>((Nx - 0.5) * dx, index_1 * dy, index_2 * dz, t)*/;
                }
            }
        }
        else if constexpr (direction == 1) // Y direction
        {
            for (Dim index_1 = 0; index_1 < Nx; ++index_1)
            {
                for (Dim index_2 = 0; index_2 < Nz; ++index_2)
                {
                    rhs.set(index_1, 0, index_2) = rhs.get(index_1, 0, index_2) /*- Real(2.0) / dy * p_boundary.first_derivative<1>(index_1 * dx, 0, index_2 * dz, t)*/;
                    rhs.set(index_1, Ny - 1, index_2) = rhs.get(index_1, Ny - 1, index_2) /*+ Real(1.0) / dy * p_boundary.first_derivative<1>(index_1 * dx, (Ny - 0.5) * dy, index_2 * dz, t)*/;
                }
            }
        }
        else if constexpr (direction == 2) // Z direction
        {
            for (Dim index_1 = 0; index_1 < Nx; ++index_1)
            {
                for (Dim index_2 = 0; index_2 < Ny; ++index_2)
                {
                    rhs.set(index_1, index_2, 0) = rhs.get(index_1, index_2, 0) /*- Real(2.0) / dz * p_boundary.first_derivative<2>(index_1 * dx, index_2 * dy, 0, t)*/;
                    rhs.set(index_1, index_2, Nz - 1) = rhs.get(index_1, index_2, Nz - 1) /*+ Real(1.0) / dz * p_boundary.first_derivative<2>(index_1 * dx, index_2 * dy, (Nz - 0.5) * dz, t)*/;
                }
            }
        }
    };

    template <Dim direction>
    void block_solver(const ScalarVariable &rhs, ScalarVariable &solution, const DimensionsHandlerScalar &dim_handler)
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
            Real t_prev = t - dt;

            // BoundaryFunctions::value is now thread-safe via thread_local parser
            solution.set(0, i, j, k) = u_boundary.value<0>(x + dx / Real(2.0), y, z, t);// - u_boundary.value<0>(x + dx / Real(2.0), y, z, t_prev);
            solution.set(1, i, j, k) = u_boundary.value<1>(x, y + dy / Real(2.0), z, t);// - u_boundary.value<1>(x, y + dy / Real(2.0), z, t_prev);
            solution.set(2, i, j, k) = u_boundary.value<2>(x, y, z + dz / Real(2.0), t);// - u_boundary.value<2>(x, y, z + dz / Real(2.0), t_prev);
        };

        if constexpr (direction == 0)
        {
            for (Dim i = 0; i < Nx; ++i)
            {
                update_bc(i, index_1, index_2);
            }
        }
        else if constexpr (direction == 1)
        {
            for (Dim j = 0; j < Ny; ++j)
                update_bc(index_1, j, index_2);
        }
        else
        {
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
                for (Dim index_2 = 0; index_2 < Nz; ++index_2)
                {
                    // on comp1 we have normal components
                    // rhs.set(direction, 0, index_1, index_2) = (u_boundary.value<direction>(0, index_1 * dy, index_2 * dz, t) - u_boundary.value<direction>(0, index_1 * dy, index_2 * dz, t - dt)) - ((u_boundary.first_derivative<1>(0, index_1 * dy, index_2 * dz, t, dy) - u_boundary.first_derivative<1>(0, index_1 * dy, index_2 * dz, t - dt, dy)) + (u_boundary.first_derivative<2>(0, index_1 * dy, index_2 * dz, t, dz) - u_boundary.first_derivative<2>(0, index_1 * dy, index_2 * dz, t - dt, dz))) * dx * Real(0.5);
                    // rhs.set(direction, Nx - 1, index_1, index_2) = u_boundary.value<direction>(Lx, index_1 * dy, index_2 * dz, t) - u_boundary.value<direction>(Lx, index_1 * dy, index_2 * dz, t - dt);
                    rhs.set(direction, 0, index_1, index_2) = (u_boundary.value<direction>(0, index_1 * dy, index_2 * dz, t)) - ((u_boundary.first_derivative<1>(0, index_1 * dy, index_2 * dz, t, dy)) + (u_boundary.first_derivative<2>(0, index_1 * dy, index_2 * dz, t, dz))) * dx * Real(0.5);
                    rhs.set(direction, Nx - 1, index_1, index_2) = u_boundary.value<direction>(Lx, index_1 * dy, index_2 * dz, t);


                    // on comp2 we have tangent components
                    // rhs.set(1, 0, index_1, index_2) = u_boundary.value<1>(0, 0.5 * dy + index_1 * dy, index_2 * dz, t) - u_boundary.value<1>(0, 0.5 * dy + index_1 * dy, index_2 * dz, t - dt);
                    // rhs.set(1, Nx - 1, index_1, index_2) = rhs.value(1, Nx - 1, index_1, index_2) + Real(2.0) * gamma_field.get(Nx - 1, index_1, index_2) / (dx * dx) * (u_boundary.value<1>(Lx, 0.5 * dy + index_1 * dy, index_2 * dz, t) - u_boundary.value<1>(Lx, 0.5 * dy + index_1 * dy, index_2 * dz, t - dt));
                    rhs.set(1, 0, index_1, index_2) = u_boundary.value<1>(0, 0.5 * dy + index_1 * dy, index_2 * dz, t);
                    rhs.set(1, Nx - 1, index_1, index_2) = rhs.value(1, Nx - 1, index_1, index_2) + Real(2.0) * gamma_field.get(Nx - 1, index_1, index_2) / (dx * dx) * (u_boundary.value<1>(Lx, 0.5 * dy + index_1 * dy, index_2 * dz, t));

                    // on comp3 we have tangent components
                    // rhs.set(2, 0, index_1, index_2) = u_boundary.value<2>(0, index_1 * dy, 0.5 * dz + index_2 * dz, t) - u_boundary.value<2>(0, index_1 * dy, 0.5 * dz + index_2 * dz, t - dt);
                    // rhs.set(2, Nx - 1, index_1, index_2) = rhs.value(2, Nx - 1, index_1, index_2) + Real(2.0) * gamma_field.get(Nx - 1, index_1, index_2) / (dx * dx) * (u_boundary.value<2>(Lx, index_1 * dy, 0.5 * dz + index_2 * dz, t) - u_boundary.value<2>(Lx, index_1 * dy, 0.5 * dz + index_2 * dz, t - dt));
                    rhs.set(2, 0, index_1, index_2) = u_boundary.value<2>(0, index_1 * dy, 0.5 * dz + index_2 * dz, t);
                    rhs.set(2, Nx - 1, index_1, index_2) = rhs.value(2, Nx - 1, index_1, index_2) + Real(2.0) * gamma_field.get(Nx - 1, index_1, index_2) / (dx * dx) * (u_boundary.value<2>(Lx, index_1 * dy, 0.5 * dz + index_2 * dz, t));
                }
            }
        }

        else if constexpr (direction == 1)
        {
            for (Dim index_1 = 0; index_1 < Nx; ++index_1)
            {
                for (Dim index_2 = 0; index_2 < Nz; ++index_2)
                {
                    // on comp2 we have normal components
                    // rhs.set(direction, index_1, 0, index_2) = (u_boundary.value<direction>(index_1 * dx, 0, index_2 * dz, t) - u_boundary.value<direction>(index_1 * dx, 0, index_2 * dz, t - dt)) - ((u_boundary.first_derivative<0>(index_1 * dx, 0, index_2 * dz, t, dx) - u_boundary.first_derivative<0>(index_1 * dx,0 , index_2 * dz, t - dt, dx)) + (u_boundary.first_derivative<2>(index_1 * dx,0, index_2 * dz, t, dz) - u_boundary.first_derivative<2>(index_1 * dx, 0, index_2 * dz, t - dt, dz))) * dy * Real(0.5);
                    // rhs.set(direction, index_1, Ny - 1, index_2) = u_boundary.value<direction>(index_1 * dx, Ly, index_2 * dz, t) - u_boundary.value<direction>(index_1 * dx, Ly, index_2 * dz, t - dt);
                    rhs.set(direction, index_1, 0, index_2) = (u_boundary.value<direction>(index_1 * dx, 0, index_2 * dz, t)) - ((u_boundary.first_derivative<0>(index_1 * dx, 0, index_2 * dz, t, dx)) + (u_boundary.first_derivative<2>(index_1 * dx,0, index_2 * dz, t, dz))) * dy * Real(0.5);
                    rhs.set(direction, index_1, Ny - 1, index_2) = u_boundary.value<direction>(index_1 * dx, Ly, index_2 * dz, t);

                    // on comp1 we have tangent components
                    // rhs.set(0, index_1, 0, index_2) = u_boundary.value<0>(0.5 * dx + index_1 * dx, 0, index_2 * dz, t) - u_boundary.value<0>(0.5 * dx + index_1 * dx, 0, index_2 * dz, t - dt);
                    // rhs.set(0, index_1, Ny - 1, index_2) = rhs.value(0, index_1, Ny - 1, index_2) + Real(2.0) * gamma_field.get(index_1, Ny - 1, index_2) / (dy * dy) * (u_boundary.value<0>(0.5 * dx + index_1 * dx, Ly, index_2 * dz, t) - u_boundary.value<0>(0.5 * dx + index_1 * dx, Ly, index_2 * dz, t - dt));
                    rhs.set(0, index_1, 0, index_2) = u_boundary.value<0>(0.5 * dx + index_1 * dx, 0, index_2 * dz, t);
                    rhs.set(0, index_1, Ny - 1, index_2) = rhs.value(0, index_1, Ny - 1, index_2) + Real(2.0) * gamma_field.get(index_1, Ny - 1, index_2) / (dy * dy) * (u_boundary.value<0>(0.5 * dx + index_1 * dx, Ly, index_2 * dz, t));

                    // on comp3 we have tangent components
                    // rhs.set(2, index_1, 0, index_2) = u_boundary.value<2>(index_1 * dx, 0, 0.5 * dz + index_2 * dz, t) - u_boundary.value<2>(index_1 * dx, 0, 0.5 * dz + index_2 * dz, t - dt);
                    // rhs.set(2, index_1, Ny - 1, index_2) = rhs.value(2, index_1, Ny - 1, index_2) + Real(2.0) * gamma_field.get(index_1, Ny - 1, index_2) / (dy * dy) * (u_boundary.value<2>(index_1 * dx, Ly, 0.5 * dz + index_2 * dz, t) - u_boundary.value<2>(index_1 * dx, Ly, 0.5 * dz + index_2 * dz, t - dt));
                    rhs.set(2, index_1, 0, index_2) = u_boundary.value<2>(index_1 * dx, 0, 0.5 * dz + index_2 * dz, t);
                    rhs.set(2, index_1, Ny - 1, index_2) = rhs.value(2, index_1, Ny - 1, index_2) + Real(2.0) * gamma_field.get(index_1, Ny - 1, index_2) / (dy * dy) * (u_boundary.value<2>(index_1 * dx, Ly, 0.5 * dz + index_2 * dz, t));
                }
            }

            // =========================================================
            // RIGHT boundary (y=Ly): lecture BCs
            // Normal comp=1 (v): Dirichlet at j=N-1 (v located at y=Ly)
            // Tangentials (u,w): ghost elimination at j=N-1
            // =========================================================
            if (comp == 1)
            {
                for (Dim index_2 = 0; index_2 < Ny; ++index_2)
                {
                    // on comp3 we have normal components
                    // rhs.set(direction, index_1, index_2, 0) = (u_boundary.value<direction>(index_1 * dx, index_2 * dy, 0, t) - u_boundary.value<direction>(index_1 * dx, index_2 * dy, 0, t - dt)) - ((u_boundary.first_derivative<0>(index_1 * dx, index_2 * dy, 0, t, dx) - u_boundary.first_derivative<0>(index_1 * dx, index_2 * dy, 0, t - dt, dx)) + (u_boundary.first_derivative<1>(index_1 * dx, index_2 * dy, 0, t, dy) - u_boundary.first_derivative<1>(index_1 * dx, index_2 * dy, 0, t - dt, dy))) * dz * Real(0.5);
                    // rhs.set(direction, index_1, index_2, Nz - 1) = u_boundary.value<direction>(index_1 * dx, index_2 * dy, Lz, t) - u_boundary.value<direction>(index_1 * dx, index_2 * dy, Lz, t - dt);
                    rhs.set(direction, index_1, index_2, 0) = (u_boundary.value<direction>(index_1 * dx, index_2 * dy, 0, t)) - ((u_boundary.first_derivative<0>(index_1 * dx, index_2 * dy, 0, t, dx)) + (u_boundary.first_derivative<1>(index_1 * dx, index_2 * dy, 0, t, dy))) * dz * Real(0.5);
                    rhs.set(direction, index_1, index_2, Nz - 1) = u_boundary.value<direction>(index_1 * dx, index_2 * dy, Lz, t);

                    // on comp1 we have tangent components
                    // rhs.set(0, index_1, index_2, 0) = (u_boundary.value<0>(0.5 * dx + index_1 * dx, index_2 * dy, 0, t) - u_boundary.value<0>(0.5 * dx + index_1 * dx, index_2 * dy, 0, t - dt));
                    // rhs.set(0, index_1, index_2, Nz - 1) = rhs.value(0, index_1, index_2, Nz - 1) + Real(2.0) * gamma_field.get(index_1, index_2, Nz - 1) / (dz * dz) * (u_boundary.value<0>(0.5 * dx + index_1 * dx, index_2 * dy, Lz, t) - u_boundary.value<0>(0.5 * dx + index_1 * dx, index_2 * dy, Lz, t - dt));
                    rhs.set(0, index_1, index_2, 0) = (u_boundary.value<0>(0.5 * dx + index_1 * dx, index_2 * dy, 0, t));
                    rhs.set(0, index_1, index_2, Nz - 1) = rhs.value(0, index_1, index_2, Nz - 1) + Real(2.0) * gamma_field.get(index_1, index_2, Nz - 1) / (dz * dz) * (u_boundary.value<0>(0.5 * dx + index_1 * dx, index_2 * dy, Lz, t));

                    // on comp2 we have tangent components
                    // rhs.set(1, index_1, index_2, 0) = (u_boundary.value<1>(index_1 * dx, 0.5 * dy + index_2 * dy, 0, t) - u_boundary.value<1>(index_1 * dx, 0.5 * dy + index_2 * dy, 0, t - dt));
                    // rhs.set(1, index_1, index_2, Nz - 1) = rhs.value(1, index_1, index_2, Nz - 1) + Real(2.0) * gamma_field.get(index_1, index_2, Nz - 1) / (dz * dz) * (u_boundary.value<1>(index_1 * dx, 0.5 * dy + index_2 * dy, Lz, t) - u_boundary.value<1>(index_1 * dx, 0.5 * dy + index_2 * dy, Lz, t - dt));
                    rhs.set(1, index_1, index_2, 0) = (u_boundary.value<1>(index_1 * dx, 0.5 * dy + index_2 * dy, 0, t));
                    rhs.set(1, index_1, index_2, Nz - 1) = rhs.value(1, index_1, index_2, Nz - 1) + Real(2.0) * gamma_field.get(index_1, index_2, Nz - 1) / (dz * dz) * (u_boundary.value<1>(index_1 * dx, 0.5 * dy + index_2 * dy, Lz, t));
                }
            }
        }
    };
    template <Dim direction>
    void block_solver(const VectorVariable &rhs, VectorVariable &solution, const DimensionsHandlerVector &dim_handler, bool use_omp = false)
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

