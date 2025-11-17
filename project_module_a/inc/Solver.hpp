#ifndef SOLVER_HPP
#define SOLVER_HPP
#include <string>
#include <cmath>
#include "Variables.hpp"
#include "ScalarVariable.hpp"
#include "VectorVariable.hpp"
#include "DimensionHandler.hpp"

class Solver
{

public:
    Dim Nx; // Grid points in x
    Dim Ny; // Grid points in y
    Dim Nz; // Grid points in z

    Real dx; // Grid spacing in x
    Real dy; // Grid spacing in y
    Real dz; // Grid spacing in z

    // Pure virtual destructor makes the class abstract
    virtual ~Solver() = 0;
    Solver(Dim Nx_, Dim Ny_, Dim Nz_, Real dx_, Real dy_, Real dz_)
        : Nx(Nx_), Ny(Ny_), Nz(Nz_),
          dx(dx_), dy(dy_), dz(dz_) {};

    template <typename StrideFunc, Dim direction>
    void solve(ScalarVariable &rhs, ScalarVariable &solution, const DimensionsHandlerScalar<StrideFunc> &dim_handler);

protected:
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

    // Helpers DIRICHLET BC
    inline Real x_at(Dim i) { return i * dx; }
    inline Real y_at(Dim j) { return j * dy; }
    inline Real z_at(Dim k) { return k * dz; }
    inline Real x_half(Dim i) { return (i + Real(0.5)) * dx; }
    inline Real y_half(Dim j) { return (j + Real(0.5)) * dy; }
    inline Real z_half(Dim k) { return (k + Real(0.5)) * dz; }
    static Real BC_u(Real, Real, Real) { return Real(1.0); }
    static Real BC_v(Real, Real, Real) { return Real(1.0); }
    static Real BC_w(Real, Real, Real) { return Real(1.0); }
};

// Definition of pure virtual destructor
inline Solver::~Solver() {}

// =============================================================================================
// =============================================================================================
// =============================================================================================
// =============================================================================================
// ====================================Pressure Solver Class====================================
// =============================================================================================
// =============================================================================================
// =============================================================================================
// =============================================================================================

class PressureSolver : public Solver
{
private:
    template <Dim direction>
    void apply_bc(ScalarVariable &rhs)
    {
        if constexpr (direction == 0) // X direction
        {
            // Implementation of Neumann boundary condition
            for (Dim index_1 = 0; index_1 < Ny; ++index_1)
            {
                for (Dim index_2 = 0; index_2 < Nz; ++index_2)
                {
                    // Lower boundary (index 0)
                    rhs.set(0, index_1, index_2) = rhs.get(0, index_1, index_2) - Real(2.0) / Nx * p_boundary.get(0, index_1, index_2); // ∂p/∂n = f at "left" boundary

                    // Upper boundary (index N1-1)
                    rhs.set(Nx - 1, index_1, index_2) = rhs.get(Nx - 1, index_1, index_2) + Real(2.0) / Nx * p_boundary.get(Nx - 1, index_1, index_2); // ∂p/∂n = f at "right" boundary
                }
            }
        }
        else if constexpr (direction == 1) // Y direction
        {
            // Implementation of Neumann boundary condition
            for (Dim index_1 = 0; index_1 < Nx; ++index_1)
            {
                for (Dim index_2 = 0; index_2 < Nz; ++index_2)
                {
                    // Lower boundary (index 0)
                    rhs.set(index_1, 0, index_2) = rhs.get(index_1, 0, index_2) - Real(2.0) / Ny * p_boundary.get(index_1, 0, index_2); // ∂p/∂n = f at "bottom" boundary

                    // Upper boundary (index N1-1)
                    rhs.set(index_1, Ny - 1, index_2) = rhs.get(index_1, Ny - 1, index_2) + Real(2.0) / Ny * p_boundary.get(index_1, Ny - 1, index_2); // ∂p/∂n = f at "top" boundary
                }
            }
        }
        else if constexpr (direction == 2) // Z direction
        {
            // Implementation of Neumann boundary condition
            for (Dim index_1 = 0; index_1 < Nx; ++index_1)
            {
                for (Dim index_2 = 0; index_2 < Ny; ++index_2)
                {
                    // Lower boundary (index 0)
                    rhs.set(index_1, index_2, 0) = rhs.get(index_1, index_2, 0) - Real(2.0) / Nz * p_boundary.get(index_1, index_2, 0); // ∂p/∂n = f at "front" boundary

                    // Upper boundary (index N1-1)
                    rhs.set(index_1, index_2, Nz - 1) = rhs.get(index_1, index_2, Nz - 1) + Real(2.0) / Nz * p_boundary.get(index_1, index_2, Nz - 1); // ∂p/∂n = f at "back" boundary
                }
            }
        }
    };

    template <Dim direction, typename StrideFunc>
    void block_solver(const ScalarVariable &rhs, ScalarVariable &solution, const DimensionsHandlerScalar<StrideFunc> &dim_handler)
    {
        // ----------------------------------------------------------------------------
        // Implementation of the block_solver method for the pressure.
        // this method decomposes the 3D problem into multiple 1D tridiagonal systems.

        std::vector<Real> a(dim_handler.N1, Real(-1.0) / (dim_handler.dN1 * dim_handler.dN1));
        std::vector<Real> b(dim_handler.N1, Real(1.0) + Real(2.0) / (dim_handler.dN1 * dim_handler.dN1));
        std::vector<Real> c(dim_handler.N1, Real(-1.0) / (dim_handler.dN1 * dim_handler.dN1));
        std::vector<Real> d(dim_handler.N1);
        std::vector<Real> x(dim_handler.N1);

        // Boundary conditions on a,b,c can be set here if needed
        a[0] = Real(0.0);
        c[0] = Real(-2.0) / (dim_handler.dN1 * dim_handler.dN1);
        b[dim_handler.N1 - 1] = Real(1.0) + Real(1.0) / (dim_handler.dN1 * dim_handler.dN1);
        c[dim_handler.N1 - 1] = Real(0.0);

        if constexpr (direction == 0) // X direction
        {
            for (Dim index_1 = 0; index_1 < Ny; ++index_1)
            {
                for (Dim index_2 = 0; index_2 < Nz; ++index_2)
                {
                    // d = rhs
                    for (Dim index_0 = 0; index_0 < Nx; ++index_0)
                    {
                        d[index_0] = p_boundary.get(index_0, index_1, index_2);
                    }

                    // Solve the tridiagonal system
                    thomas_algorithm(a, b, c, d, x);

                    // Store the solution
                    for (Dim index_0 = 0; index_0 < Nx; ++index_0)
                    {
                        solution.set(index_0, index_1, index_2) = x[index_0];
                    }
                }
            }
        }
        else if constexpr (direction == 1) // Y direction
        {
            for (Dim index_1 = 0; index_1 < Nx; ++index_1)
            {
                for (Dim index_2 = 0; index_2 < Nz; ++index_2)
                {
                    // d = rhs
                    for (Dim index_0 = 0; index_0 < Ny; ++index_0)
                    {
                        d[index_0] = p_boundary.get(index_0, index_1, index_2);
                    }

                    // Solve the tridiagonal system
                    thomas_algorithm(a, b, c, d, x);

                    // Store the solution
                    for (Dim index_0 = 0; index_0 < Ny; ++index_0)
                    {
                        solution.set(index_0, index_1, index_2) = x[index_0];
                    }
                }
            }
        }
        else if constexpr (direction == 2) // Z direction
        {
            for (Dim index_1 = 0; index_1 < Nx; ++index_1)
            {
                for (Dim index_2 = 0; index_2 < Ny; ++index_2)
                {
                    // d = rhs
                    for (Dim index_0 = 0; index_0 < Nz; ++index_0)
                    {
                        d[index_0] = p_boundary.get(index_0, index_1, index_2);
                    }

                    // Solve the tridiagonal system
                    thomas_algorithm(a, b, c, d, x);

                    // Store the solution
                    for (Dim index_0 = 0; index_0 < Nz; ++index_0)
                    {
                        solution.set(index_0, index_1, index_2) = x[index_0];
                    }
                }
            }
        }
    };
    ScalarVariable &p_boundary;

public:
    PressureSolver(Dim Nx_, Dim Ny_, Dim Nz_, Real dx_, Real dy_, Real dz_, ScalarVariable &p_boundary_)
        : Solver(Nx_, Ny_, Nz_, dx_, dy_, dz_), p_boundary(p_boundary_)
    {
    }
    template <typename StrideFunc, Dim direction>
    void solve_pressure(ScalarVariable &rhs, ScalarVariable &solution, const DimensionsHandlerScalar<StrideFunc> &dim_handler)
    {
        apply_bc<direction>(rhs); // <-- CORRECTED: Removed dim_handler
        block_solver<direction, StrideFunc>(rhs, solution, dim_handler);
    };
    ScalarVariable &set_p_boundary() { return p_boundary; }
};
// =============================================================================================
// =============================================================================================
// =============================================================================================
// =============================================================================================
// ====================================Velocity Solver Class====================================
// =============================================================================================
// =============================================================================================
// =============================================================================================
// =============================================================================================

class VelocitySolver : public Solver
{
private:
    ScalarVariable &gamma_field;

    template <Dim direction>
    bool is_known_face(Dim index_1, Dim index_2, Dim component) const
    {
        if constexpr (direction == 0)
        {
            if (component == 0)
            {
                return (index_1 == 0 || index_2 == 0);
            }
            else if (component == 1)
            {
                return (index_1 == Ny - 1 || index_2 == 0);
            }
            else if (component == 2)
            {
                return (index_1 == 0 || index_2 == Nz - 1);
            }
        }
        else if constexpr (direction == 1)
        {
            if (component == 0)
            {
                return (index_1 == Nx - 1 || index_2 == 0);
            }
            else if (component == 1)
            {
                return (index_1 == 0 || index_2 == 0);
            }
            else if (component == 2)
            {
                return (index_1 == 0 || index_2 == Nz - 1);
            }
        }
        else // direction == 2
        {
            if (component == 0)
            {
                return (index_1 == Nx - 1 || index_2 == 0);
            }
            else if (component == 1)
            {
                return (index_1 == 0 || index_2 == Ny - 1);
            }
            else if (component == 2)
            {
                return (index_1 == 0 || index_2 == 0);
            }
        }
    }

    template <Dim direction, typename StrideFunc>
    bool handle_known_face(const DimensionsHandlerVector<StrideFunc> &dim_handler, VectorVariable &solution, Dim index_1, Dim index_2, Dim component)
    {
        if (!is_known_face<direction>(index_1, index_2, component))
        {
            return false;
        }

        Dim Comp1 = dim_handler.Comp1;
        Dim Comp2 = dim_handler.Comp2;
        Dim Comp3 = dim_handler.Comp3;

        if constexpr (direction == 0)
        {
            for (Dim index_0 = 0; index_0 < Nx; ++index_0)
            {
                solution.set(Comp1, index_0, index_1, index_2) = u_boundary.value(Comp1, index_0, index_1, index_2);
                solution.set(Comp2, index_0, index_1, index_2) = u_boundary.value(Comp2, index_0, index_1, index_2);
                solution.set(Comp3, index_0, index_1, index_2) = u_boundary.value(Comp3, index_0, index_1, index_2);
            }
        }
        else if constexpr (direction == 1)
        {
            for (Dim index_0 = 0; index_0 < Ny; ++index_0)
            {
                solution.set(Comp1, index_1, index_0, index_2) = u_boundary.value(Comp1, index_1, index_0, index_2);
                solution.set(Comp2, index_1, index_0, index_2) = u_boundary.value(Comp2, index_1, index_0, index_2);
                solution.set(Comp3, index_1, index_0, index_2) = u_boundary.value(Comp3, index_1, index_0, index_2);
            }
        }
        else // direction == 2
        {
            for (Dim index_0 = 0; index_0 < Nz; ++index_0)
            {
                solution.set(Comp1, index_1, index_2, index_0) = u_boundary.value(Comp1, index_1, index_2, index_0);
                solution.set(Comp2, index_1, index_2, index_0) = u_boundary.value(Comp2, index_1, index_2, index_0);
                solution.set(Comp3, index_1, index_2, index_0) = u_boundary.value(Comp3, index_1, index_2, index_0);
            }
        }

        return true;
    }

    template <Dim direction, typename StrideFunc>
    void block_solver(const VectorVariable &rhs, VectorVariable &solution, const DimensionsHandlerVector<StrideFunc> &dim_handler)
    {
        Dim N1 = dim_handler.N1;
        Dim N2 = dim_handler.N2;
        Dim N3 = dim_handler.N3;
        Real dN1 = dim_handler.dN1;
        auto stride_func = dim_handler.stride;

        std::vector<Real> a(N1, Real(0.0));
        std::vector<Real> b(N1, Real(1.0));
        std::vector<Real> c(N1, Real(0.0));
        std::vector<Real> d(N1, Real(0.0));
        std::vector<Real> x(N1, Real(0.0));

        // Boundary conditions are already set

        // Here we consider only the even indices in 2nd direction
        // where normal components are considered
        if constexpr (direction == 0) // X direction
        {
            // Comp1 = 0 (x-component, normal on x-boundaries)
            // Comp2 = 1 (y-component, tangent on x-boundaries)
            // Comp3 = 2 (z-component, tangent on x-boundaries)
            Dim Comp1 = dim_handler.Comp1; // 0 (x)
            Dim Comp2 = dim_handler.Comp2; // 1 (y)
            Dim Comp3 = dim_handler.Comp3; // 2 (z)

            for (Dim index_1 = 0; index_1 < Ny; ++index_1)
            {
                for (Dim index_2 = 0; index_2 < Nz; ++index_2)
                {
                    if (!handle_known_face<direction>(dim_handler, solution, index_1, index_2, Comp1))
                    {

                        // Solve for Comp1 (x-component, normal on x-boundaries)
                        for (Dim index_0 = 1; index_0 < Nx - 1; ++index_0)
                        {
                            d[index_0] = rhs.value(Comp1, index_0, index_1, index_2);
                            Real gamma_val = -gamma_field.get(index_0, index_1, index_2);
                            a[index_0] = gamma_val / (dx * dx);
                            b[index_0] = 1.0f - (2.0f * gamma_val) / (dx * dx);
                            c[index_0] = gamma_val / (dx * dx);
                        }
                        // Left boundary: identity row (incompressibility handled in rhs by apply_bc)
                        a[0] = Real(0.0);
                        b[0] = Real(1.0);
                        c[0] = Real(0.0);
                        d[0] = rhs.value(Comp1, 0, index_1, index_2);
                        // Right boundary: identity row
                        a[Nx - 1] = Real(0.0);
                        b[Nx - 1] = Real(1.0);
                        c[Nx - 1] = Real(0.0);
                        d[Nx - 1] = rhs.value(Comp1, Nx - 1, index_1, index_2);
                        thomas_algorithm(a, b, c, d, x);
                        for (Dim index_0 = 0; index_0 < Nx; ++index_0)
                        {
                            solution.set(Comp1, index_0, index_1, index_2) = x[index_0];
                        }
                    }
                    // Solve for Comp2 (y-component, tangent on x-boundaries)
                    if (!handle_known_face<direction>(dim_handler, solution, index_1, index_2, Comp2))
                    {
                        for (Dim index_0 = 1; index_0 < Nx - 1; ++index_0)
                        {
                            d[index_0] = rhs.value(Comp2, index_0, index_1, index_2);
                            Real gamma_val = -gamma_field.get(index_0, index_1, index_2);
                            a[index_0] = gamma_val / (dx * dx);
                            b[index_0] = 1.0f - (2.0f * gamma_val) / (dx * dx);
                            c[index_0] = gamma_val / (dx * dx);
                        }
                        // Left boundary: identity row
                        a[0] = Real(0.0);
                        b[0] = Real(1.0);
                        c[0] = Real(0.0);
                        d[0] = rhs.value(Comp2, 0, index_1, index_2);
                        // Right boundary: ghost node elimination
                        Real gamma_N = -gamma_field.get(Nx - 1, index_1, index_2);
                        Real c_val = gamma_N / (dx * dx);
                        a[Nx - 1] = gamma_N / (dx * dx);
                        b[Nx - 1] = Real(1.0) - (Real(2.0) * gamma_N) / (dx * dx) - c_val; // b - c
                        c[Nx - 1] = Real(0.0);
                        d[Nx - 1] = rhs.value(Comp2, Nx - 1, index_1, index_2);
                        thomas_algorithm(a, b, c, d, x);
                        for (Dim index_0 = 0; index_0 < Nx; ++index_0)
                        {
                            solution.set(Comp2, index_0, index_1, index_2) = x[index_0];
                        }
                    }
                    // Solve for Comp3 (z-component, tangent on x-boundaries)
                    if (!handle_known_face<direction>(dim_handler, solution, index_1, index_2, Comp3))
                    {
                        for (Dim index_0 = 1; index_0 < Nx - 1; ++index_0)
                        {
                            d[index_0] = rhs.value(Comp3, index_0, index_1, index_2);
                            Real gamma_val = -gamma_field.get(index_0, index_1, index_2);
                            a[index_0] = gamma_val / (dx * dx);
                            b[index_0] = 1.0f - (2.0f * gamma_val) / (dx * dx);
                            c[index_0] = gamma_val / (dx * dx);
                        }
                        // Left boundary: identity row
                        a[0] = Real(0.0);
                        b[0] = Real(1.0);
                        c[0] = Real(0.0);
                        d[0] = rhs.value(Comp3, 0, index_1, index_2);
                        // Right boundary: ghost node elimination
                        gamma_N = -gamma_field.get(Nx - 1, index_1, index_2);
                        c_val = gamma_N / (dx * dx);
                        a[Nx - 1] = gamma_N / (dx * dx);
                        b[Nx - 1] = Real(1.0) - (Real(2.0) * gamma_N) / (dx * dx) - c_val; // b - c
                        c[Nx - 1] = Real(0.0);
                        d[Nx - 1] = rhs.value(Comp3, Nx - 1, index_1, index_2);
                        thomas_algorithm(a, b, c, d, x);
                        for (Dim index_0 = 0; index_0 < Nx; ++index_0)
                        {
                            solution.set(Comp3, index_0, index_1, index_2) = x[index_0];
                        }
                    }
                }
            }
        }
        else if constexpr (direction == 1) // Y direction
        {
            // Comp1 = 1 (y-component, normal on y-boundaries)
            // Comp2 = 0 (x-component, tangent on y-boundaries)
            // Comp3 = 2 (z-component, tangent on y-boundaries)
            Dim Comp1 = dim_handler.Comp1; // 1 (y)
            Dim Comp2 = dim_handler.Comp2; // 0 (x)
            Dim Comp3 = dim_handler.Comp3; // 2 (z)

            for (Dim index_1 = 0; index_1 < Nx; ++index_1)
            {
                for (Dim index_2 = 0; index_2 < Nz; ++index_2)
                {
                    if (!handle_known_face<direction>(dim_handler, solution, index_1, index_2, Comp1))
                    {
                    // Solve for Comp1 (y-component, normal on y-boundaries)
                    // Note: index_0 is y-index, index_1 is x-index, index_2 is z-index
                        for (Dim index_0 = 1; index_0 < Ny - 1; ++index_0)
                        {
                            d[index_0] = rhs.value(Comp1, index_1, index_0, index_2);
                            Real gamma_val = -gamma_field.get(index_1, index_0, index_2);
                            a[index_0] = gamma_val / (dy * dy);
                            b[index_0] = 1.0f - (2.0f * gamma_val) / (dy * dy);
                            c[index_0] = gamma_val / (dy * dy);
                        }
                        // Left boundary: identity row (incompressibility handled in rhs by apply_bc)
                        a[0] = Real(0.0);
                        b[0] = Real(1.0);
                        c[0] = Real(0.0);
                        d[0] = rhs.value(Comp1, index_1, 0, index_2);
                        // Right boundary: identity row
                        a[Ny - 1] = Real(0.0);
                        b[Ny - 1] = Real(1.0);
                        c[Ny - 1] = Real(0.0);
                        d[Ny - 1] = rhs.value(Comp1, index_1, Ny - 1, index_2);
                        thomas_algorithm(a, b, c, d, x);
                        for (Dim index_0 = 0; index_0 < Ny; ++index_0)
                        {
                            solution.set(Comp1, index_1, index_0, index_2) = x[index_0];
                        }
                    }
                    if (!handle_known_face<direction>(dim_handler, solution, index_1, index_2, Comp2))
                    {
                        // Solve for Comp2 (x-component, tangent on y-boundaries)
                        for (Dim index_0 = 1; index_0 < Ny - 1; ++index_0)
                        {
                            d[index_0] = rhs.value(Comp2, index_1, index_0, index_2);
                            Real gamma_val = -gamma_field.get(index_1, index_0, index_2);
                            a[index_0] = gamma_val / (dy * dy);
                            b[index_0] = 1.0f - (2.0f * gamma_val) / (dy * dy);
                            c[index_0] = gamma_val / (dy * dy);
                        }
                        // Left boundary: identity row
                        a[0] = Real(0.0);
                        b[0] = Real(1.0);
                        c[0] = Real(0.0);
                        d[0] = rhs.value(Comp2, index_1, 0, index_2);
                        // Right boundary: ghost node elimination
                        Real gamma_N = -gamma_field.get(index_1, Ny - 1, index_2);
                        Real c_val = gamma_N / (dy * dy);
                        a[Ny - 1] = gamma_N / (dy * dy);
                        b[Ny - 1] = Real(1.0) - (Real(2.0) * gamma_N) / (dy * dy) - c_val; // b - c
                        c[Ny - 1] = Real(0.0);
                        d[Ny - 1] = rhs.value(Comp2, index_1, Ny - 1, index_2);
                        thomas_algorithm(a, b, c, d, x);
                        for (Dim index_0 = 0; index_0 < Ny; ++index_0)
                        {
                            solution.set(Comp2, index_1, index_0, index_2) = x[index_0];
                        }
                    }
                    if (!handle_known_face<direction>(dim_handler, solution, index_1, index_2, Comp3))
                    {
                        // Solve for Comp3 (z-component, tangent on y-boundaries)
                        for (Dim index_0 = 1; index_0 < Ny - 1; ++index_0)
                        {
                            d[index_0] = rhs.value(Comp3, index_1, index_0, index_2);
                            Real gamma_val = -gamma_field.get(index_1, index_0, index_2);
                            a[index_0] = gamma_val / (dy * dy);
                            b[index_0] = 1.0f - (2.0f * gamma_val) / (dy * dy);
                            c[index_0] = gamma_val / (dy * dy);
                        }
                        // Left boundary: identity row
                        a[0] = Real(0.0);
                        b[0] = Real(1.0);
                        c[0] = Real(0.0);
                        d[0] = rhs.value(Comp3, index_1, 0, index_2);
                        // Right boundary: ghost node elimination
                        gamma_N = -gamma_field.get(index_1, Ny - 1, index_2);
                        c_val = gamma_N / (dy * dy);
                        a[Ny - 1] = gamma_N / (dy * dy);
                        b[Ny - 1] = Real(1.0) - (Real(2.0) * gamma_N) / (dy * dy) - c_val; // b - c
                        c[Ny - 1] = Real(0.0);
                        d[Ny - 1] = rhs.value(Comp3, index_1, Ny - 1, index_2);
                        thomas_algorithm(a, b, c, d, x);
                        for (Dim index_0 = 0; index_0 < Ny; ++index_0)
                        {
                            solution.set(Comp3, index_1, index_0, index_2) = x[index_0];
                        }
                    }
                }
            }
        }
        else if constexpr (direction == 2) // Z direction
        {
            // Comp1 = 2 (z-component, normal on z-boundaries)
            // Comp2 = 0 (x-component, tangent on z-boundaries)
            // Comp3 = 1 (y-component, tangent on z-boundaries)
            Dim Comp1 = dim_handler.Comp1; // 2 (z)
            Dim Comp2 = dim_handler.Comp2; // 0 (x)
            Dim Comp3 = dim_handler.Comp3; // 1 (y)

            for (Dim index_1 = 0; index_1 < Nx; ++index_1)
            {
                for (Dim index_2 = 0; index_2 < Ny; ++index_2)
                {
                    if (!handle_known_face<direction>(dim_handler, solution, index_1, index_2, Comp1))
                    {

                        // Solve for Comp1 (z-component, normal on z-boundaries)
                        // Note: index_0 is z-index, index_1 is x-index, index_2 is y-index
                        for (Dim index_0 = 1; index_0 < Nz - 1; ++index_0)
                        {
                            d[index_0] = rhs.value(Comp1, index_1, index_2, index_0);
                            Real gamma_val = -gamma_field.get(index_1, index_2, index_0);
                            a[index_0] = gamma_val / (dz * dz);
                            b[index_0] = 1.0f - (2.0f * gamma_val) / (dz * dz);
                            c[index_0] = gamma_val / (dz * dz);
                        }
                        // Left boundary: identity row (incompressibility handled in rhs by apply_bc)
                        a[0] = Real(0.0);
                        b[0] = Real(1.0);
                        c[0] = Real(0.0);
                        d[0] = rhs.value(Comp1, index_1, index_2, 0);
                        // Right boundary: identity row
                        a[Nz - 1] = Real(0.0);
                        b[Nz - 1] = Real(1.0);
                        c[Nz - 1] = Real(0.0);
                        d[Nz - 1] = rhs.value(Comp1, index_1, index_2, Nz - 1);
                        thomas_algorithm(a, b, c, d, x);
                        for (Dim index_0 = 0; index_0 < Nz; ++index_0)
                        {
                            solution.set(Comp1, index_1, index_2, index_0) = x[index_0];
                        }
                    }
                    if (!handle_known_face<direction>(dim_handler, solution, index_1, index_2, Comp2))
                    {
                        // Solve for Comp2 (x-component, tangent on z-boundaries)
                        for (Dim index_0 = 1; index_0 < Nz - 1; ++index_0)
                        {
                            d[index_0] = rhs.value(Comp2, index_1, index_2, index_0);
                            Real gamma_val = -gamma_field.get(index_1, index_2, index_0);
                            a[index_0] = gamma_val / (dz * dz);
                            b[index_0] = 1.0f - (2.0f * gamma_val) / (dz * dz);
                            c[index_0] = gamma_val / (dz * dz);
                        }
                        // Left boundary: identity row
                        a[0] = Real(0.0);
                        b[0] = Real(1.0);
                        c[0] = Real(0.0);
                        d[0] = rhs.value(Comp2, index_1, index_2, 0);
                        // Right boundary: ghost node elimination
                        Real gamma_N = -gamma_field.get(index_1, index_2, Nz - 1);
                        Real c_val = gamma_N / (dz * dz);
                        a[Nz - 1] = gamma_N / (dz * dz);
                        b[Nz - 1] = Real(1.0) - (Real(2.0) * gamma_N) / (dz * dz) - c_val; // b - c
                        c[Nz - 1] = Real(0.0);
                        d[Nz - 1] = rhs.value(Comp2, index_1, index_2, Nz - 1);
                        thomas_algorithm(a, b, c, d, x);
                        for (Dim index_0 = 0; index_0 < Nz; ++index_0)
                        {
                            solution.set(Comp2, index_1, index_2, index_0) = x[index_0];
                        }
                    }
                    if (!handle_known_face<direction>(dim_handler, solution, index_1, index_2, Comp3))
                    {
                        // Solve for Comp3 (y-component, tangent on z-boundaries)
                        for (Dim index_0 = 1; index_0 < Nz - 1; ++index_0)
                        {
                            d[index_0] = rhs.value(Comp3, index_1, index_2, index_0);
                            Real gamma_val = -gamma_field.get(index_1, index_2, index_0);
                            a[index_0] = gamma_val / (dz * dz);
                            b[index_0] = 1.0f - (2.0f * gamma_val) / (dz * dz);
                            c[index_0] = gamma_val / (dz * dz);
                        }
                        // Left boundary: identity row
                        a[0] = Real(0.0);
                        b[0] = Real(1.0);
                        c[0] = Real(0.0);
                        d[0] = rhs.value(Comp3, index_1, index_2, 0);
                        // Right boundary: ghost node elimination
                        gamma_N = -gamma_field.get(index_1, index_2, Nz - 1);
                        c_val = gamma_N / (dz * dz);
                        a[Nz - 1] = gamma_N / (dz * dz);
                        b[Nz - 1] = Real(1.0) - (Real(2.0) * gamma_N) / (dz * dz) - c_val; // b - c
                        c[Nz - 1] = Real(0.0);
                        d[Nz - 1] = rhs.value(Comp3, index_1, index_2, Nz - 1);
                        thomas_algorithm(a, b, c, d, x);
                        for (Dim index_0 = 0; index_0 < Nz; ++index_0)
                        {
                            solution.set(Comp3, index_1, index_2, index_0) = x[index_0];
                        }
                    }
                }
            }
        }
    };

    template <typename StrideFunc, Dim direction>
    void apply_bc(VectorVariable &rhs, const DimensionsHandlerVector<StrideFunc> &dim_handler)
    {

        // Here we consider only the even indices in 2nd direction
        // where we have normal components on even 3rd direction and tangent components on odd 3rd direction

        if constexpr (direction == 0)
        {
            for (Dim index_1 = 0; index_1 < Ny; ++index_1)
            {
                for (Dim index_2 = 0; index_2 < Nz; ++index_2)
                {
                    // on comp1 we have normal components
                    rhs.set(direction, 0, index_1, index_2) = u_boundary.value(direction, 0, index_1, index_2) - (u_boundary.first_derivative(1, 1, 0, index_1, index_2) + u_boundary.first_derivative(2, 2, 0, index_1, index_2)) * dx * Real(0.5);
                    rhs.set(direction, Nx - 1, index_1, index_2) = u_boundary.value(direction, Nx - 1, index_1, index_2);

                    // on comp2 we have tangent components
                    rhs.set(1, 0, index_1, index_2) = u_boundary.value(1, 0, index_1, index_2);
                    rhs.set(1, Nx - 1, index_1, index_2) = rhs.value(1, Nx - 1, index_1, index_2) + Real(2.0) * gamma_field.get(Nx - 1, index_1, index_2) / (dx * dx) * u_boundary.value(1, Nx - 1, index_1, index_2);

                    // on comp3 we have tangent components
                    rhs.set(2, 0, index_1, index_2) = u_boundary.value(2, 0, index_1, index_2);
                    rhs.set(2, Nx - 1, index_1, index_2) = rhs.value(2, Nx - 1, index_1, index_2) + Real(2.0) * gamma_field.get(Nx - 1, index_1, index_2) / (dx * dx) * u_boundary.value(2, Nx - 1, index_1, index_2);
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
                    rhs.set(direction, index_1, 0, index_2) = u_boundary.value(direction, index_1, 0, index_2) - (u_boundary.first_derivative(0, 0, index_1, 0, index_2) + u_boundary.first_derivative(2, 2, index_1, 0, index_2)) * dy * Real(0.5);
                    rhs.set(direction, index_1, Ny - 1, index_2) = u_boundary.value(direction, index_1, Ny - 1, index_2);

                    // on comp1 we have tangent components
                    rhs.set(0, index_1, 0, index_2) = u_boundary.value(0, index_1, 0, index_2);
                    rhs.set(0, index_1, Ny - 1, index_2) = rhs.value(0, index_1, Ny - 1, index_2) + Real(2.0) * gamma_field.get(index_1, Ny - 1, index_2) / (dy * dy) * u_boundary.value(0, index_1, Ny - 1, index_2);

                    // on comp3 we have tangent components
                    rhs.set(2, index_1, 0, index_2) = u_boundary.value(2, index_1, 0, index_2);
                    rhs.set(2, index_1, Ny - 1, index_2) = rhs.value(2, index_1, Ny - 1, index_2) + Real(2.0) * gamma_field.get(index_1, Ny - 1, index_2) / (dy * dy) * u_boundary.value(2, index_1, Ny - 1, index_2);
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
                    rhs.set(direction, index_1, index_2, 0) = u_boundary.value(direction, index_1, index_2, 0) - (u_boundary.first_derivative(0, 0, index_1, index_2, 0) + u_boundary.first_derivative(1, 1, index_1, index_2, 0)) * dz * Real(0.5);
                    rhs.set(direction, index_1, index_2, Nz - 1) = u_boundary.value(direction, index_1, index_2, Nz - 1);

                    // on comp1 we have tangent components
                    rhs.set(0, index_1, index_2, 0) = u_boundary.value(0, index_1, index_2, 0);
                    rhs.set(0, index_1, index_2, Nz - 1) = rhs.value(0, index_1, index_2, Nz - 1) + Real(2.0) * gamma_field.get(index_1, index_2, Nz - 1) / (dz * dz) * u_boundary.value(0, index_1, index_2, Nz - 1);

                    // on comp2 we have tangent components
                    rhs.set(1, index_1, index_2, 0) = u_boundary.value(1, index_1, index_2, 0);
                    rhs.set(1, index_1, index_2, Nz - 1) = rhs.value(1, index_1, index_2, Nz - 1) + Real(2.0) * gamma_field.get(index_1, index_2, Nz - 1) / (dz * dz) * u_boundary.value(1, index_1, index_2, Nz - 1);
                }
            }
        }
    };

    VectorVariable &u_boundary;

public:
    VelocitySolver(Dim Nx_, Dim Ny_, Dim Nz_, Real dx_, Real dy_, Real dz_, ScalarVariable &gam, VectorVariable &u_bnd)
        : Solver(Nx_, Ny_, Nz_, dx_, dy_, dz_), gamma_field(gam), u_boundary(u_bnd)
    {
    }

    template <typename StrideFunc, Dim direction>
    void solve(VectorVariable &rhs, VectorVariable &solution, const DimensionsHandlerVector<StrideFunc> &dim_handler)
    {
        apply_bc<StrideFunc, direction>(rhs, dim_handler);
        block_solver<direction, StrideFunc>(rhs, solution, dim_handler);
    };
    void set_gamma(ScalarVariable &g) { gamma_field = g; }
    VectorVariable &set_u_boundary() { return u_boundary; }
};
#endif // SOLVER_HPP