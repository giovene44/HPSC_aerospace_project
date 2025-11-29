#ifndef SOLVER_HPP
#define SOLVER_HPP
#include <string>
#include <cmath>
#include <functional> // For std::function/lambdas
#include "Variables.hpp"
#include "ScalarVariable.hpp"
#include "VectorVariable.hpp"
#include "DimensionHandler.hpp"
#include "BoundaryFunctions.hpp"

class Solver
{

public:
    // Pure virtual destructor makes the class abstract
    virtual ~Solver() = 0;
    Solver(Dim Nx_, Dim Ny_, Dim Nz_, Real dx_, Real dy_, Real dz_, Real dt_)
        : Nx(Nx_), Ny(Ny_), Nz(Nz_),
          dx(dx_), dy(dy_), dz(dz_), dt(dt_) { t = dt_; }; // solve doesn't solve for t=0 called firstly at t=dt

    template <typename StrideFunc, Dim direction>
    void solve(ScalarVariable &rhs, ScalarVariable &solution, const DimensionsHandlerScalar<StrideFunc> &dim_handler);
    void advance_time()
    {
        t += dt;
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
                    rhs.set(0, index_1, index_2) = rhs.get(0, index_1, index_2) - Real(2.0) / dx * p_boundary.value<0>(0, index_1 * dy, index_2 * dz, t);
                    rhs.set(Nx - 1, index_1, index_2) = rhs.get(Nx - 1, index_1, index_2) + Real(1.0) / dx * p_boundary.value<0>((Nx - 0.5) * dx, index_1 * dy, index_2 * dz, t);
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

    template <Dim direction, typename StrideFunc>
    void block_solver(const ScalarVariable &rhs, ScalarVariable &solution, const DimensionsHandlerScalar<StrideFunc> &dim_handler)
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

    /**
     * @brief Applies the Pressure Poisson matrix A to a scalar field x to compute RHS = A*x.
     * Mimics the coefficients used in block_solver for consistency check.
     */
    template <Dim direction, typename StrideFunc>
    void apply_matrix_operator(const ScalarVariable &x_scalar, ScalarVariable &rhs_scalar, ScalarVariable &true_rhs, const DimensionsHandlerScalar<StrideFunc> &dim_handler)
    {
        // 1. Determine Dimensions and Grid Spacing based on direction
        Dim N = (direction == 0) ? Nx : ((direction == 1) ? Ny : Nz);
        Real h = (direction == 0) ? dx : ((direction == 1) ? dy : dz);
        Real h2 = h * h;

        // Precompute Coefficients
        Real coeff = 1.0 / h2;

        // Standard Interior Stencil: -1/h^2, 1 + 2/h^2, -1/h^2
        // Wait! In your block_solver:
        // a = -1/h^2
        // b = 1 + 2/h^2
        // c = -1/h^2
        Real a_int = -coeff;
        Real b_int = 1.0 + 2.0 * coeff;
        Real c_int = -coeff;

        // Boundary Start (i=0):
        // In block_solver: a=0, c = -2/h^2 (due to p_-1 = p_1)
        // b is standard (1 + 2/h^2)
        Real b_start = b_int;
        Real c_start = -2.0 * coeff;

        // Boundary End (i=N-1):
        // In block_solver: c=0, b = 1 + 1/h^2 (due to p_N+1 = p_N? Check logic)
        // Your code: b[N-1] = 1.0 + 1.0/(h*h)
        // This implies the BC was p_N+1 = p_N (Homogeneous Neumann at right wall?)
        // Standard Neumann is p_N+1 = p_N-1 (Centered) or p_N (Forward)
        // Let's match YOUR code exactly:
        Real a_end = a_int;             // -1/h^2
        Real b_end = 1.0 + 1.0 * coeff; // Matches your code: Real(1.0) + Real(1.0)/...

        // 2. Loop Limits
        Dim Outer1 = (direction == 0) ? Ny : ((direction == 1) ? Nx : Nx);
        Dim Outer2 = (direction == 0) ? Nz : ((direction == 1) ? Nz : Ny);

        for (Dim i1 = 0; i1 < Outer1; ++i1)
        {
            for (Dim i2 = 0; i2 < Outer2; ++i2)
            {
                // Lambdas for access
                auto get_val = [&](Dim i)
                {
                    if constexpr (direction == 0)
                        return x_scalar.get(i, i1, i2);
                    else if constexpr (direction == 1)
                        return x_scalar.get(i1, i, i2);
                    else
                        return x_scalar.get(i1, i2, i);
                };

                auto set_rhs = [&](Dim i, Real val)
                {
                    if constexpr (direction == 0)
                        rhs_scalar.set(i, i1, i2) = val;
                    else if constexpr (direction == 1)
                        rhs_scalar.set(i1, i, i2) = val;
                    else
                        rhs_scalar.set(i1, i2, i) = val;
                };

                for (Dim i = 0; i < N; ++i)
                {
                    Real val = 0.0;

                    if (i == 0)
                    {
                        // Boundary Start (Neumann Left: p_-1 = p_1)
                        // Row 0: b*p_0 + c*p_1 (where c is doubled)
                        val = b_start * get_val(0) + c_start * get_val(1);
                    }
                    else if (i == N - 1)
                    {
                        // Boundary End
                        // Row N-1: a*p_{N-2} + b*p_{N-1}
                        val = a_end * get_val(i - 1) + b_end * get_val(i);
                    }
                    else
                    {
                        // Interior
                        // a*p_{i-1} + b*p_i + c*p_{i+1}
                        val = a_int * get_val(i - 1) + b_int * get_val(i) + c_int * get_val(i + 1);
                    }

                    set_rhs(i, val);
                }
            }
        }
    }

    BoundaryFunctions &p_boundary;

    PressureSolver(Dim Nx_, Dim Ny_, Dim Nz_, Real dx_, Real dy_, Real dz_, Real dt_, BoundaryFunctions &p_boundary_)
        : Solver(Nx_, Ny_, Nz_, dx_, dy_, dz_, dt_), p_boundary(p_boundary_) {}

    template <typename StrideFunc, Dim direction>
    void solve_pressure(ScalarVariable &rhs, ScalarVariable &solution, const DimensionsHandlerScalar<StrideFunc> &dim_handler)
    {
        apply_bc<direction>(rhs);
        block_solver<direction, StrideFunc>(rhs, solution, dim_handler);
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

    /**
     * @brief Applies the system matrix A to a vector x to compute RHS = A*x.
     * This mimics the implicit operator coefficients used in block_solver.
     */
    /**
     * @brief Applies the system matrix A to a vector x to compute RHS = A*x.
     * Handles known faces exactly like block_solver to ensure consistent verification.
     */
    template <Dim direction, typename StrideFunc>
    void apply_matrix_operator(const VectorVariable &x_vec, VectorVariable &rhs_vec, VectorVariable &true_rhs, const DimensionsHandlerVector<StrideFunc> &dim_handler)
    {
        Dim N = (direction == 0) ? Nx : ((direction == 1) ? Ny : Nz);
        Real h = (direction == 0) ? dx : ((direction == 1) ? dy : dz);

        Dim Comp1 = dim_handler.Comp1; // Normal component
        Dim Comp2 = dim_handler.Comp2; // Tangent 1
        Dim Comp3 = dim_handler.Comp3; // Tangent 2

        Dim Outer1 = (direction == 0) ? Ny : ((direction == 1) ? Nx : Nx);
        Dim Outer2 = (direction == 0) ? Nz : ((direction == 1) ? Nz : Ny);

        for (Dim i1 = 0; i1 < Outer1; ++i1)
        {
            for (Dim i2 = 0; i2 < Outer2; ++i2)
            {
                // Helper lambdas
                auto get_val = [&](Dim comp, Dim i)
                {
                    if constexpr (direction == 0)
                        return x_vec.value(comp, i, i1, i2);
                    else if constexpr (direction == 1)
                        return x_vec.value(comp, i1, i, i2);
                    else
                        return x_vec.value(comp, i1, i2, i);
                };

                auto set_rhs = [&](Dim comp, Dim i, Real val)
                {
                    if constexpr (direction == 0)
                        rhs_vec.set(comp, i, i1, i2) = val;
                    else if constexpr (direction == 1)
                        rhs_vec.set(comp, i1, i, i2) = val;
                    else
                        rhs_vec.set(comp, i1, i2, i) = val;
                };

                auto get_gamma_val = [&](Dim i)
                {
                    if constexpr (direction == 0)
                        return gamma_field.get(i, i1, i2);
                    else if constexpr (direction == 1)
                        return gamma_field.get(i1, i, i2);
                    else
                        return gamma_field.get(i1, i2, i);
                };

                auto get_true_rhs = [&](Dim comp, Dim i)
                {
                    if constexpr (direction == 0)
                        return true_rhs.value(comp, i, i1, i2);
                    else if constexpr (direction == 1)
                        return true_rhs.value(comp, i1, i, i2);
                    else
                        return true_rhs.value(comp, i1, i2, i);
                };

                // --- COMPONENT 1 (Normal) ---
                if (is_known_face<direction>(i1, i2, Comp1))
                {
                    // Known Face -> Identity Matrix row: RHS = x
                    for (Dim i = 0; i < N; ++i)
                        set_rhs(Comp1, i, get_val(Comp1, i));
                }
                else
                {
                    for (Dim i = 0; i < N; ++i)
                    {
                        if (i == 0 || i == N - 1) // Boundaries
                        {
                            set_rhs(Comp1, i, get_val(Comp1, i));
                        }
                        else // Internal
                        {
                            Real coeff = get_gamma_val(i) / (h * h);
                            Real val = (-coeff) * get_val(Comp1, i - 1) +
                                       (1.0 + 2.0 * coeff) * get_val(Comp1, i) +
                                       (-coeff) * get_val(Comp1, i + 1);
                            set_rhs(Comp1, i, val);
                        }
                    }
                }

                // --- COMPONENT 2 (Tangent 1) ---
                if (is_known_face<direction>(i1, i2, Comp2))
                {
                    for (Dim i = 0; i < N; ++i)
                        set_rhs(Comp2, i, get_val(Comp2, i));
                }
                else
                {
                    for (Dim i = 0; i < N; ++i)
                    {
                        if (i == 0) // Left Boundary (Identity)
                        {
                            set_rhs(Comp2, i, get_true_rhs(Comp2, i));
                        }
                        else if (i == N - 1) // Right Boundary (Modified Neumann)
                        {
                            set_rhs(Comp2, i, get_true_rhs(Comp2, i));
                        }
                        else // Internal
                        {
                            Real coeff = get_gamma_val(i) / (h * h);
                            Real val = (-coeff) * get_val(Comp2, i - 1) +
                                       (1.0 + 2.0 * coeff) * get_val(Comp2, i) +
                                       (-coeff) * get_val(Comp2, i + 1);
                            set_rhs(Comp2, i, val);
                        }
                    }
                }

                // --- COMPONENT 3 (Tangent 2) ---
                if (is_known_face<direction>(i1, i2, Comp3))
                {
                    for (Dim i = 0; i < N; ++i)
                        set_rhs(Comp3, i, get_val(Comp3, i));
                }
                else
                {
                    for (Dim i = 0; i < N; ++i)
                    {
                        if (i == 0) // Left Boundary (Identity)
                        {
                            set_rhs(Comp3, i, get_true_rhs(Comp3, i));
                        }
                        else if (i == N - 1) // Right Boundary (Modified Neumann)
                        {
                            set_rhs(Comp3, i, get_true_rhs(Comp3, i));
                        }
                        else // Internal
                        {
                            Real coeff = get_gamma_val(i) / (h * h);
                            Real val = (-coeff) * get_val(Comp3, i - 1) +
                                       (1.0 + 2.0 * coeff) * get_val(Comp3, i) +
                                       (-coeff) * get_val(Comp3, i + 1);
                            set_rhs(Comp3, i, val);
                        }
                    }
                }
            }
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
        else if constexpr (direction == 1)
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
        throw std::invalid_argument("Invalid component");
    }

    template <Dim direction, typename StrideFunc>
    bool handle_known_face(const DimensionsHandlerVector<StrideFunc> &dim_handler, VectorVariable &solution, Dim index_1, Dim index_2, Dim component)
    {
        if (!is_known_face<direction>(index_1, index_2, component))
            return false;

        Dim Comp1 = dim_handler.Comp1;
        Dim Comp2 = dim_handler.Comp2;
        Dim Comp3 = dim_handler.Comp3;
        const Real t_prev = (t - dt < 0.0f) ? 0.0f : t - dt;

        auto update_bc = [&](Dim i, Dim j, Dim k)
        {
            Real x = i * dx, y = j * dy, z = k * dz;
            if (t == 0.0f)
            {
                solution.set(Comp1, i, j, k) = u_boundary.value<0>(x + dx / Real(2.0), y, z, t);
                solution.set(Comp2, i, j, k) = u_boundary.value<1>(x, y + dy / Real(2.0), z, t);
                solution.set(Comp3, i, j, k) = u_boundary.value<2>(x, y, z + dz / Real(2.0), t);
                return;
            }
            solution.set(Comp1, i, j, k) = u_boundary.value<0>(x + dx / Real(2.0), y, z, t) - u_boundary.value<0>(x + dx / Real(2.0), y, z, t_prev);
            solution.set(Comp2, i, j, k) = u_boundary.value<1>(x, y + dy / Real(2.0), z, t) - u_boundary.value<1>(x, y + dy / Real(2.0), z, t_prev);
            solution.set(Comp3, i, j, k) = u_boundary.value<2>(x, y, z + dz / Real(2.0), t) - u_boundary.value<2>(x, y, z + dz / Real(2.0), t_prev);
        };

        if constexpr (direction == 0)
        {
            for (Dim i = 0; i < Nx; ++i)
            {
                update_bc(i, index_1, index_2);
                if (index_1 == 0 && index_2 == 0)
                {
                }
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
                    rhs.set(direction, 0, index_1, index_2) = (u_boundary.value<direction>(0, index_1 * dy, index_2 * dz, t) - u_boundary.value<direction>(0, index_1 * dy, index_2 * dz, t - dt)) - ((u_boundary.first_derivative<1>(0, index_1 * dy, index_2 * dz, t, dy) - u_boundary.first_derivative<1>(0, index_1 * dy, index_2 * dz, t - dt, dy)) + (u_boundary.first_derivative<2>(0, index_1 * dy, index_2 * dz, t, dz) - u_boundary.first_derivative<2>(0, index_1 * dy, index_2 * dz, t - dt, dz))) * dx * Real(0.5);
                    rhs.set(direction, Nx - 1, index_1, index_2) = u_boundary.value<direction>(Lx, index_1 * dy, index_2 * dz, t) - u_boundary.value<direction>(Lx, index_1 * dy, index_2 * dz, t - dt);

                    // on comp2 we have tangent components
                    rhs.set(1, 0, index_1, index_2) = u_boundary.value<1>(0, 0.5 * dy + index_1 * dy, index_2 * dz, t) - u_boundary.value<1>(0, 0.5 * dy + index_1 * dy, index_2 * dz, t - dt);
                    rhs.set(1, Nx - 1, index_1, index_2) = rhs.value(1, Nx - 1, index_1, index_2) + Real(2.0) * gamma_field.get(Nx - 1, index_1, index_2) / (dx * dx) * (u_boundary.value<1>(Lx, 0.5 * dy + index_1 * dy, index_2 * dz, t) - u_boundary.value<1>(Lx, 0.5 * dy + index_1 * dy, index_2 * dz, t - dt));

                    // on comp3 we have tangent components
                    rhs.set(2, 0, index_1, index_2) = u_boundary.value<2>(0, index_1 * dy, 0.5 * dz + index_2 * dz, t) - u_boundary.value<2>(0, index_1 * dy, 0.5 * dz + index_2 * dz, t - dt);
                    rhs.set(2, Nx - 1, index_1, index_2) = rhs.value(2, Nx - 1, index_1, index_2) + Real(2.0) * gamma_field.get(Nx - 1, index_1, index_2) / (dx * dx) * (u_boundary.value<2>(Lx, index_1 * dy, 0.5 * dz + index_2 * dz, t) - u_boundary.value<2>(Lx, index_1 * dy, 0.5 * dz + index_2 * dz, t - dt));
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
                    rhs.set(direction, index_1, 0, index_2) = (u_boundary.value<direction>(index_1 * dx, 0, index_2 * dz, t) - u_boundary.value<direction>(index_1 * dx, 0, index_2 * dz, t - dt)) - ((u_boundary.first_derivative<0>(index_1 * dx, 0, index_2 * dz, t, dx) - u_boundary.first_derivative<0>(index_1 * dx, 0, index_2 * dz, t - dt, dx)) + (u_boundary.first_derivative<2>(index_1 * dx, 0, index_2 * dz, t, dz) - u_boundary.first_derivative<2>(index_1 * dx, 0, index_2 * dz, t - dt, dz))) * dy * Real(0.5);
                    rhs.set(direction, index_1, Ny - 1, index_2) = u_boundary.value<direction>(index_1 * dx, Ly, index_2 * dz, t) - u_boundary.value<direction>(index_1 * dx, Ly, index_2 * dz, t - dt);

                    // on comp1 we have tangent components
                    rhs.set(0, index_1, 0, index_2) = u_boundary.value<0>(0.5 * dx + index_1 * dx, 0, index_2 * dz, t) - u_boundary.value<0>(0.5 * dx + index_1 * dx, 0, index_2 * dz, t - dt);
                    rhs.set(0, index_1, Ny - 1, index_2) = rhs.value(0, index_1, Ny - 1, index_2) + Real(2.0) * gamma_field.get(index_1, Ny - 1, index_2) / (dy * dy) * (u_boundary.value<0>(0.5 * dx + index_1 * dx, Ly, index_2 * dz, t) - u_boundary.value<0>(0.5 * dx + index_1 * dx, Ly, index_2 * dz, t - dt));
                    // on comp3 we have tangent components
                    rhs.set(2, index_1, 0, index_2) = u_boundary.value<2>(index_1 * dx, 0, 0.5 * dz + index_2 * dz, t) - u_boundary.value<2>(index_1 * dx, 0, 0.5 * dz + index_2 * dz, t - dt);
                    rhs.set(2, index_1, Ny - 1, index_2) = rhs.value(2, index_1, Ny - 1, index_2) + Real(2.0) * gamma_field.get(index_1, Ny - 1, index_2) / (dy * dy) * (u_boundary.value<2>(index_1 * dx, Ly, 0.5 * dz + index_2 * dz, t) - u_boundary.value<2>(index_1 * dx, Ly, 0.5 * dz + index_2 * dz, t - dt));
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
                    rhs.set(direction, index_1, index_2, 0) = (u_boundary.value<direction>(index_1 * dx, index_2 * dy, 0, t) - u_boundary.value<direction>(index_1 * dx, index_2 * dy, 0, t - dt)) - ((u_boundary.first_derivative<0>(index_1 * dx, index_2 * dy, 0, t, dx) - u_boundary.first_derivative<0>(index_1 * dx, index_2 * dy, 0, t - dt, dx)) + (u_boundary.first_derivative<1>(index_1 * dx, index_2 * dy, 0, t, dy) - u_boundary.first_derivative<1>(index_1 * dx, index_2 * dy, 0, t - dt, dy))) * dz * Real(0.5);
                    rhs.set(direction, index_1, index_2, Nz - 1) = (u_boundary.value<direction>(index_1 * dx, index_2 * dy, dz + (Nz - 1) * dz, t) - u_boundary.value<direction>(index_1 * dx, index_2 * dy, dz + (Nz - 1) * dz, t - dt));

                    // on comp1 we have tangent components
                    rhs.set(0, index_1, index_2, 0) = (u_boundary.value<0>(0.5 * dx + index_1 * dx, index_2 * dy, 0, t) - u_boundary.value<0>(0.5 * dx + index_1 * dx, index_2 * dy, 0, t - dt));
                    rhs.set(0, index_1, index_2, Nz - 1) = rhs.value(0, index_1, index_2, Nz - 1) + Real(2.0) * gamma_field.get(index_1, index_2, Nz - 1) / (dz * dz) * (u_boundary.value<0>(0.5 * dx + index_1 * dx, index_2 * dy, Lz, t) - u_boundary.value<0>(0.5 * dx + index_1 * dx, index_2 * dy, Lz, t - dt));

                    // on comp2 we have tangent components
                    rhs.set(1, index_1, index_2, 0) = (u_boundary.value<1>(index_1 * dx, 0.5 * dy + index_2 * dy, 0, t) - u_boundary.value<1>(index_1 * dx, 0.5 * dy + index_2 * dy, 0, t - dt));
                    rhs.set(1, index_1, index_2, Nz - 1) = rhs.value(1, index_1, index_2, Nz - 1) + Real(2.0) * gamma_field.get(index_1, index_2, Nz - 1) / (dz * dz) * (u_boundary.value<1>(index_1 * dx, 0.5 * dy + index_2 * dy, Lz, t) - u_boundary.value<1>(index_1 * dx, 0.5 * dy + index_2 * dy, Lz, t - dt));
                }
            }
        }
    };
    template <Dim direction, typename StrideFunc>
    void block_solver(const VectorVariable &rhs, VectorVariable &solution, const DimensionsHandlerVector<StrideFunc> &dim_handler)
    {
        Dim N = (direction == 0) ? Nx : ((direction == 1) ? Ny : Nz);
        Real h = (direction == 0) ? dx : ((direction == 1) ? dy : dz);

        Dim Comp1 = dim_handler.Comp1;
        Dim Comp2 = dim_handler.Comp2;
        Dim Comp3 = dim_handler.Comp3;

        std::vector<Real> a(N), b(N), c(N), d(N), x(N);

        Dim Outer1 = (direction == 0) ? Ny : ((direction == 1) ? Nx : Nx);
        Dim Outer2 = (direction == 0) ? Nz : ((direction == 1) ? Nz : Ny);

        for (Dim i1 = 0; i1 < Outer1; ++i1)
        {
            for (Dim i2 = 0; i2 < Outer2; ++i2)
            {
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
                if (!handle_known_face<direction>(dim_handler, solution, i1, i2, Comp1))
                {
                    setup_TDMA_internal(N, h, a, b, c, d, [&](Dim i)
                                        { return get_rhs_comp(Comp1, i); }, get_gamma);

                    a[0] = 0.0;
                    b[0] = 1.0;
                    c[0] = 0.0;
                    d[0] = get_rhs_comp(Comp1, 0);
                    a[N - 1] = 0.0;
                    b[N - 1] = 1.0;
                    c[N - 1] = 0.0;
                    d[N - 1] = get_rhs_comp(Comp1, N - 1);
                    thomas_algorithm(a, b, c, d, x);
                    for (Dim i = 0; i < N; ++i)
                        set_sol_comp(Comp1, i, x[i]);
                }

                if (!handle_known_face<direction>(dim_handler, solution, i1, i2, Comp2))
                {
                    setup_TDMA_internal(N, h, a, b, c, d, [&](Dim i)
                                        { return get_rhs_comp(Comp2, i); }, get_gamma);
                    a[0] = 0.0;
                    b[0] = 1.0;
                    c[0] = 0.0;
                    d[0] = get_rhs_comp(Comp2, 0);
                    Real gamma_N = get_gamma(N - 1);
                    Real coeff = gamma_N / (h * h);
                    a[N - 1] = -coeff;
                    b[N - 1] = (1.0f + 2.0f * coeff) - (-coeff);
                    c[N - 1] = 0.0;
                    d[N - 1] = get_rhs_comp(Comp2, N - 1);
                    thomas_algorithm(a, b, c, d, x);
                    for (Dim i = 0; i < N; ++i)
                        set_sol_comp(Comp2, i, x[i]);
                }

                // Comp3 (Tangent)
                if (!handle_known_face<direction>(dim_handler, solution, i1, i2, Comp3))
                {
                    setup_TDMA_internal(N, h, a, b, c, d, [&](Dim i)
                                        { return get_rhs_comp(Comp3, i); }, get_gamma);
                    a[0] = 0.0;
                    b[0] = 1.0;
                    c[0] = 0.0;
                    d[0] = get_rhs_comp(Comp3, 0);
                    Real gamma_N = get_gamma(N - 1);
                    Real coeff = gamma_N / (h * h);
                    a[N - 1] = -coeff;
                    b[N - 1] = (1.0f + 2.0f * coeff) - (-coeff);
                    c[N - 1] = 0.0;
                    d[N - 1] = get_rhs_comp(Comp3, N - 1);
                    thomas_algorithm(a, b, c, d, x);
                    for (Dim i = 0; i < N; ++i)
                        set_sol_comp(Comp3, i, x[i]);
                }
            }
        }
    }

    VelocitySolver(Dim Nx_, Dim Ny_, Dim Nz_, Real dx_, Real dy_, Real dz_, Real dt_, ScalarVariable &gam, BoundaryFunctions &u_bnd)
        : Solver(Nx_, Ny_, Nz_, dx_, dy_, dz_, dt_), gamma_field(gam), u_boundary(u_bnd) {}

    template <typename StrideFunc, Dim direction>
    void solve(VectorVariable &rhs, VectorVariable &solution, const DimensionsHandlerVector<StrideFunc> &dim_handler)
    {
        apply_bc<direction>(rhs);
        block_solver<direction, StrideFunc>(rhs, solution, dim_handler);
        advance_time();
    };
    void set_gamma(ScalarVariable &g) { gamma_field = g; }
    BoundaryFunctions &set_u_boundary() { return u_boundary; }
};
#endif // SOLVER_HPP
