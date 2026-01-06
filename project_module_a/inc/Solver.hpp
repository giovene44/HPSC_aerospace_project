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
                    rhs.set(0, index_1, index_2) = rhs.get(0, index_1, index_2) + Real(2.0) * dx * (-Real(1.0) / (dx * dx)) * p_boundary.value<0>(0, index_1 * dy, index_2 * dz, t);
                    rhs.set(Nx - 1, index_1, index_2) = rhs.get(Nx - 1, index_1, index_2) - dx * (-Real(1.0) / (dx * dx)) * p_boundary.value<0>((Nx - 0.5) * dx, index_1 * dy, index_2 * dz, t);
                }
            }
        }
        else if constexpr (direction == 1) // Y direction
        {
            for (Dim index_1 = 0; index_1 < Nx; ++index_1)
            {
                for (Dim index_2 = 0; index_2 < Nz; ++index_2)
                {
                    rhs.set(index_1, 0, index_2) = rhs.get(index_1, 0, index_2) - Real(2.0) / dy * p_boundary.value<1>(index_1 * dx, 0, index_2 * dz, t);
                    rhs.set(index_1, Ny - 1, index_2) = rhs.get(index_1, Ny - 1, index_2) + Real(1.0) / dy * p_boundary.value<1>(index_1 * dx, (Ny - 0.5) * dy, index_2 * dz, t);
                }
            }
        }
        else if constexpr (direction == 2) // Z direction
        {
            for (Dim index_1 = 0; index_1 < Nx; ++index_1)
            {
                for (Dim index_2 = 0; index_2 < Ny; ++index_2)
                {
                    rhs.set(index_1, index_2, 0) = rhs.get(index_1, index_2, 0) - Real(2.0) / dz * p_boundary.value<2>(index_1 * dx, index_2 * dy, 0, t);
                    rhs.set(index_1, index_2, Nz - 1) = rhs.get(index_1, index_2, Nz - 1) + Real(1.0) / dz * p_boundary.value<2>(index_1 * dx, index_2 * dy, (Nz - 0.5) * dz, t);
                }
            }
        }
    };

    template <Dim direction>
    void block_solver(const ScalarVariable &rhs, ScalarVariable &solution, const DimensionsHandlerScalar &dim_handler)
    {
        std::vector<Real> a(dim_handler.N1, Real(-1.0) / (dim_handler.dN1 * dim_handler.dN1));
        std::vector<Real> b(dim_handler.N1, Real(1.0) + (Real(2.0) / (dim_handler.dN1 * dim_handler.dN1)));
        std::vector<Real> c(dim_handler.N1, Real(-1.0) / (dim_handler.dN1 * dim_handler.dN1));
        std::vector<Real> d(dim_handler.N1);
        std::vector<Real> x(dim_handler.N1);

        a[0] = Real(0.0);
        c[0] = Real(-2.0) / (dim_handler.dN1 * dim_handler.dN1);
        b[dim_handler.N1 - 1] = Real(1.0) + Real(1.0) / (dim_handler.dN1 * dim_handler.dN1);
        c[dim_handler.N1 - 1] = Real(0.0);

        if constexpr (direction == 0)
        {
            for (Dim index_1 = 0; index_1 < Ny; ++index_1)
            {
                for (Dim index_2 = 0; index_2 < Nz; ++index_2)
                {
                    for (Dim index_0 = 0; index_0 < Nx; ++index_0)
                        d[index_0] = rhs.get(index_0, index_1, index_2);
                    thomas_algorithm(a, b, c, d, x);
                    for (Dim index_0 = 0; index_0 < Nx; ++index_0)
                        solution.set(index_0, index_1, index_2) = x[index_0];
                }
            }
        }
        else if constexpr (direction == 1)
        {
            for (Dim index_1 = 0; index_1 < Nx; ++index_1)
            {
                for (Dim index_2 = 0; index_2 < Nz; ++index_2)
                {
                    for (Dim index_0 = 0; index_0 < Ny; ++index_0)
                        d[index_0] = rhs.get(index_1, index_0, index_2);
                    thomas_algorithm(a, b, c, d, x);
                    for (Dim index_0 = 0; index_0 < Ny; ++index_0)
                        solution.set(index_1, index_0, index_2) = x[index_0];
                }
            }
        }
        else if constexpr (direction == 2)
        {
            for (Dim index_1 = 0; index_1 < Nx; ++index_1)
            {
                for (Dim index_2 = 0; index_2 < Ny; ++index_2)
                {
                    for (Dim index_0 = 0; index_0 < Nz; ++index_0)
                        d[index_0] = rhs.get(index_1, index_2, index_0);
                    thomas_algorithm(a, b, c, d, x);
                    for (Dim index_0 = 0; index_0 < Nz; ++index_0)
                        solution.set(index_1, index_2, index_0) = x[index_0];
                }
            }
        }
    }

    BoundaryFunctions &p_boundary;

    PressureSolver(Dim Nx_, Dim Ny_, Dim Nz_, Real dx_, Real dy_, Real dz_, Real dt_, BoundaryFunctions &p_boundary_)
        : Solver(Nx_, Ny_, Nz_, dx_, dy_, dz_, dt_), p_boundary(p_boundary_) {}

    template <Dim direction>
    void solve_pressure(ScalarVariable &rhs, ScalarVariable &solution, const DimensionsHandlerScalar &dim_handler)
    {
        apply_bc<direction>(rhs);
        block_solver<direction>(rhs, solution, dim_handler);
        advance_time();
    };
    BoundaryFunctions &set_p_boundary() { return p_boundary; }
};

// =============================================================================================
// ====================================Velocity Solver Class====================================
// =============================================================================================

class VelocitySolver : public Solver
{
public:
    ScalarVariable &gamma_field;
    BoundaryFunctions &u_boundary;

    // --- HELPER: Computes a,b,c,d for INTERNAL points (1 to N-2) ---
    // Coefficients matched to Professor's slides: a = -gamma/h^2, b = 1 + 2*gamma/h^2, c = -gamma/h^2
    template <typename RhsGetter, typename GammaGetter>
    void setup_TDMA_internal(
        Dim N, Real h,
        std::vector<Real> &a, std::vector<Real> &b, std::vector<Real> &c, std::vector<Real> &d,
        RhsGetter get_rhs, GammaGetter get_gamma)
    {
        Real h2 = h * h;
        // Iterate over internal points only
        for (Dim i = 1; i < N - 1; ++i)
        {
            Real gamma_val = get_gamma(i); // Assumes gamma is positive
            Real coeff = gamma_val / h2;

            d[i] = get_rhs(i);

            // STABLE COEFFICIENTS for (I - gamma*L) u = RHS
            a[i] = -coeff;
            b[i] = 1.0f + 2.0f * coeff;
            c[i] = -coeff;
        }
    }

    template <Dim direction>
    bool is_known_face(Dim index_1, Dim index_2, Dim component) const
    {
        if constexpr (direction == 0)
        {
            if (component == 0)
                return (index_1 == 0 || index_2 == 0);
            if (component == 1)
                return (index_1 == Ny - 1 || index_2 == 0);
            if (component == 2)
                return (index_1 == 0 || index_2 == Nz - 1);
        }
        else if constexpr (direction == 1) // Sweep Y-direction
        {
            if (component == 0)
                return (index_1 == Nx - 1 || index_2 == 0);
            if (component == 1)
                return (index_1 == 0 || index_2 == 0);
            if (component == 2)
                return (index_1 == 0 || index_2 == Nz - 1);
        }
        else
        { // direction == 2
            if (component == 0)
                return (index_1 == Nx - 1 || index_2 == 0);
            if (component == 1)
                return (index_1 == 0 || index_2 == Ny - 1);
            if (component == 2)
                return (index_1 == 0 || index_2 == 0);
        }
        return false;
    }
    template <Dim direction>
    bool handle_known_face(VectorVariable &solution, Dim index_1, Dim index_2, Dim component)
    {
        if (!is_known_face<direction>(index_1, index_2, component))
            return false;

        auto update_bc = [&](Dim i, Dim j, Dim k)
        {
            Real x = i * dx, y = j * dy, z = k * dz;
            if (t == 0.0f)
            {
                printf("IF YOU SEE THIS MESSAGE IN VelocitySolver::handle_known_face THEN SOMETHING IS WRONG\n");
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
                update_bc(index_1, index_2, k);
        }

        return true;
    }
    template <Dim direction>
    void apply_bc(VectorVariable &rhs)
    {
        // Domain lengths:
        Real Lx = dx * (Nx - 0.5);
        Real Ly = dy * (Ny - 0.5);
        Real Lz = dz * (Nz - 0.5);

        if constexpr (direction == 0)
        {
            for (Dim index_1 = 0; index_1 < Ny; ++index_1)
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
        }
        else if constexpr (direction == 2)
        {
            for (Dim index_1 = 0; index_1 < Nx; ++index_1)
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
        Dim N = (direction == 0) ? Nx : ((direction == 1) ? Ny : Nz);
        Real h = (direction == 0) ? dx : ((direction == 1) ? dy : dz);

        Dim Comp1 = dim_handler.Comp1;
        Dim Comp2 = dim_handler.Comp2;
        Dim Comp3 = dim_handler.Comp3;

        std::vector<Real> a(N), b(N), c(N), d(N), x(N);

        Dim Outer1 = dim_handler.N2;
        Dim Outer2 = dim_handler.N3;

        auto worker = [&](Dim i1, Dim i2)
        {
            std::vector<Real> a_loc = a, b_loc = b, c_loc = c, d_loc = d, x_loc = x;

            auto get_gamma = [&](Dim i)
            {
                if constexpr (direction == 0)
                    return gamma_field.get(i, i1, i2);
                else if constexpr (direction == 1)
                    return gamma_field.get(i1, i, i2);
                else
                    return gamma_field.get(i1, i2, i);
            };
            auto get_rhs_comp = [&](Dim comp, Dim i)
            {
                if constexpr (direction == 0)
                    return rhs.value(comp, i, i1, i2);
                else if constexpr (direction == 1)
                    return rhs.value(comp, i1, i, i2);
                else
                    return rhs.value(comp, i1, i2, i);
            };
            auto set_sol_comp = [&](Dim comp, Dim i, Real val)
            {
                if constexpr (direction == 0)
                    solution.set(comp, i, i1, i2) = val;
                else if constexpr (direction == 1)
                    solution.set(comp, i1, i, i2) = val;
                else
                    solution.set(comp, i1, i2, i) = val;
            };

            // Comp1 (Normal)
            if (!handle_known_face<direction>(solution, i1, i2, Comp1))
            {
                setup_TDMA_internal(N, h, a_loc, b_loc, c_loc, d_loc, [&](Dim i)
                                    { return get_rhs_comp(Comp1, i); }, get_gamma);

                a_loc[0] = 0.0;
                b_loc[0] = 1.0;
                c_loc[0] = 0.0;
                d_loc[0] = get_rhs_comp(Comp1, 0);
                a_loc[N - 1] = 0.0;
                b_loc[N - 1] = 1.0;
                c_loc[N - 1] = 0.0;
                d_loc[N - 1] = get_rhs_comp(Comp1, N - 1);
                thomas_algorithm(a_loc, b_loc, c_loc, d_loc, x_loc);
                for (Dim i = 0; i < N; ++i)
                    set_sol_comp(Comp1, i, x_loc[i]);
            }

            // Comp2 (Tangent)
            if (!handle_known_face<direction>(solution, i1, i2, Comp2))
            {
                setup_TDMA_internal(N, h, a_loc, b_loc, c_loc, d_loc, [&](Dim i)
                                    { return get_rhs_comp(Comp2, i); }, get_gamma);
                a_loc[0] = 0.0;
                b_loc[0] = 1.0;
                c_loc[0] = 0.0;
                d_loc[0] = get_rhs_comp(Comp2, 0);
                Real gamma_N = get_gamma(N - 1);
                Real coeff = gamma_N / (h * h);
                a_loc[N - 1] = -coeff;
                b_loc[N - 1] = (1.0f + 2.0f * coeff) - (-coeff);
                c_loc[N - 1] = 0.0;
                d_loc[N - 1] = get_rhs_comp(Comp2, N - 1);
                thomas_algorithm(a_loc, b_loc, c_loc, d_loc, x_loc);
                for (Dim i = 0; i < N; ++i)
                    set_sol_comp(Comp2, i, x_loc[i]);
            }

            // Comp3 (Tangent)
            if (!handle_known_face<direction>(solution, i1, i2, Comp3))
            {
                setup_TDMA_internal(N, h, a_loc, b_loc, c_loc, d_loc, [&](Dim i)
                                    { return get_rhs_comp(Comp3, i); }, get_gamma);
                a_loc[0] = 0.0;
                b_loc[0] = 1.0;
                c_loc[0] = 0.0;
                d_loc[0] = get_rhs_comp(Comp3, 0);
                Real gamma_N = get_gamma(N - 1);
                Real coeff = gamma_N / (h * h);
                a_loc[N - 1] = -coeff;
                b_loc[N - 1] = (1.0f + 2.0f * coeff) - (-coeff);
                c_loc[N - 1] = 0.0;
                d_loc[N - 1] = get_rhs_comp(Comp3, N - 1);
                thomas_algorithm(a_loc, b_loc, c_loc, d_loc, x_loc);
                for (Dim i = 0; i < N; ++i)
                    set_sol_comp(Comp3, i, x_loc[i]);
            }
        };

#ifdef _OPENMP
        if (use_omp)
        {
            if (DEBUG_BLOCK)
            {
                int max_threads = omp_get_max_threads();
                printf("[OMP] block_solver<%d>: using up to %d threads\n", int(direction), max_threads);
            }
#pragma omp parallel for collapse(2) default(none) shared(Outer1, Outer2, worker)
            for (Dim i2 = 0; i2 < Outer2; ++i2)
                for (Dim i1 = 0; i1 < Outer1; ++i1)
                    worker(i1, i2);
            return;
        }

#endif

        for (Dim i2 = 0; i2 < Outer2; ++i2)
            for (Dim i1 = 0; i1 < Outer1; ++i1)
                worker(i1, i2);
    }

    VelocitySolver(Dim Nx_, Dim Ny_, Dim Nz_, Real dx_, Real dy_, Real dz_, Real dt_, ScalarVariable &gam, BoundaryFunctions &u_bnd)
        : Solver(Nx_, Ny_, Nz_, dx_, dy_, dz_, dt_), gamma_field(gam), u_boundary(u_bnd) {}

    template <Dim direction>
    void solve(VectorVariable &rhs, VectorVariable &solution, const DimensionsHandlerVector &dim_handler, bool use_omp = false)
    {
        apply_bc<direction>(rhs);
        block_solver<direction>(rhs, solution, dim_handler, use_omp);
    };
    void set_gamma(ScalarVariable &g) { gamma_field = g; }
    BoundaryFunctions &set_u_boundary() { return u_boundary; }
};
#endif // SOLVER_HPP

