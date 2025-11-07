#ifndef SOLVER_HPP
#define SOLVER_HPP
#include <string>
#include <cmath>
#include "ScalarVariable.hpp"
#include "VectorVariable.hpp"

template <typename StrideFunction>
struct DimensionsHandler
{
    Dim N1;
    Dim N2;
    Dim N3;
    Real dN1;
    StrideFunction stride;

    DimensionsHandler(Dim N1, Dim N2, Dim N3, Real dN1, StrideFunction stride)
        : N1(N1), N2(N2), N3(N3), dN1(dN1), stride(stride)
    {
    }
};

class Solver
{

public:
    Dim Nx; // Grid points in x
    Dim Ny; // Grid points in y
    Dim Nz; // Grid points in z

    Real dx; // Grid spacing in x
    Real dy; // Grid spacing in y
    Real dz; // Grid spacing in z

    struct StrideX
    {
        Dim Nx, Ny;
        // Sweep along x -> vary i -> fix (j,k)
        Dim operator()(Dim j, Dim k) const
        {
            return j * Nx + k * Nx * Ny;
        }
    };

    struct StrideY
    {
        Dim Nx, Ny;
        // Sweep along y -> vary j -> fix (i,k)
        Dim operator()(Dim i, Dim k) const
        {
            return i + k * Nx * Ny;
        }
    };

    struct StrideZ
    {
        Dim Nx;
        // Sweep along z -> vary k -> fix (i,j)
        Dim operator()(Dim i, Dim j) const
        {
            return i + j * Nx;
        }
    };

    DimensionsHandler<StrideX> dim_hand_x;
    DimensionsHandler<StrideY> dim_hand_y;
    DimensionsHandler<StrideZ> dim_hand_z;

    virtual ~Solver() = default;
    Solver(Dim Nx_, Dim Ny_, Dim Nz_, Real dx_, Real dy_, Real dz_, ScalarVariable gam)
        : Nx(Nx_), Ny(Ny_), Nz(Nz_),
          dx(dx_), dy(dy_), dz(dz_),
          dim_hand_x(Nx, Ny, Nz, dx, StrideX{Ny_, Nz_}),
          dim_hand_y(Ny, Nx, Nz, dy, StrideY{Nx_, Nz_}),
          dim_hand_z(Nz, Nx, Ny, dz, StrideZ{Nx_}), gamma_field(gam) {
          };

    virtual void solve_x_dir(ScalarVariable &, ScalarVariable &) = 0;
    virtual void solve_y_dir(ScalarVariable &, ScalarVariable &) = 0;
    virtual void solve_z_dir(ScalarVariable &, ScalarVariable &) = 0;

    void set_gamma(const ScalarVariable &g) { gamma_field = g; }

private:
    void thomas_algorithm(const std::vector<float> &a, const std::vector<float> &b, const std::vector<float> &c, const std::vector<float> &rhs, std::vector<float> &x)
    {
        int n = rhs.size();
        std::vector<float> c_prime(c.size(), 0.0);
        std::vector<float> rhs_prime(n, 0.0);

        c_prime[0] = c[0] / b[0];
        rhs_prime[0] = rhs[0] / b[0];

        for (int i = 1; i < n; ++i)
        {
            float m = 1.0 / (b[i] - a[i] * c_prime[i - 1]);
            c_prime[i] = c[i] * m;
            rhs_prime[i] = (rhs[i] - a[i] * rhs_prime[i - 1]) * m;
        }

        x[n - 1] = rhs_prime[n - 1];

        for (int i = n - 2; i >= 0; --i)
        {
            x[i] = rhs_prime[i] - c_prime[i] * x[i + 1];
        }
    }

protected:
    ScalarVariable gamma_field;

    virtual void apply_bc_x_dir(ScalarVariable &) = 0;
    virtual void apply_bc_y_dir(ScalarVariable &) = 0;
    virtual void apply_bc_z_dir(ScalarVariable &) = 0;

    template <typename StrideFunc>
    void block_solver_press(const ScalarVariable &rhs, ScalarVariable &solution, const DimensionsHandler<StrideFunc> &dim_hand)
    {
        // ----------------------------------------------------------------------------
        // Implementation of the block_solver method for the pressure.
        // this method decomposes the 3D problem into multiple 1D tridiagonal systems.

        Dim N1 = dim_hand.N1;
        Dim N2 = dim_hand.N2;
        Dim N3 = dim_hand.N3;
        Real dN1 = dim_hand.dN1;
        auto stride_func = dim_hand.stride;

        std::vector<Real> a(N1, Real(-1.0) / (dN1 * dN1));
        std::vector<Real> b(N1, Real(1.0) + Real(2.0) / (dN1 * dN1));
        std::vector<Real> c(N1, Real(-1.0) / (dN1 * dN1));
        std::vector<Real> d(N1);
        std::vector<Real> x(N1);

        // Boundary conditions on a,b,c can be set here if needed
        a[0] = 0.0f;
        c[0] = -2.0f / (dN1 * dN1);
        b[N1 - 1] = 1.0f + 1.0f / (dN1 * dN1);
        c[N1 - 1] = 0.0f;

        for (Dim inedx_1 = 0; inedx_1 < N2; ++inedx_1)
        {
            for (Dim inedx_2 = 0; inedx_2 < N3; ++inedx_2)
            {

                Dim stride = stride_func(inedx_1, inedx_2);

                // d = rhs
                for (Dim index_0 = 0; index_0 < N1; ++index_0)
                {
                    d[index_0] = rhs.get(stride + index_0);
                }

                // Solve the tridiagonal system
                thomas_algorithm(a, b, c, d, x);

                // Store the solution
                for (Dim index_0 = 0; index_0 < N1; ++index_0)
                {
                    solution.set(stride + index_0) = x[index_0];
                }
            }
        }
    };

    template <typename StrideFunc>
    void block_solver_mom(const ScalarVariable &rhs, ScalarVariable &solution, const DimensionsHandler<StrideFunc> &dim_hand)
    {
        Dim N1 = dim_hand.N1;
        Dim N2 = dim_hand.N2;
        Dim N3 = dim_hand.N3;
        Real dN1 = dim_hand.dN1;
        auto stride_func = dim_hand.stride;

        std::vector<Real> a(N1, Real(0.0));
        std::vector<Real> b(N1, Real(1.0));
        std::vector<Real> c(N1, Real(0.0));
        std::vector<Real> d(N1, Real(0.0));
        std::vector<Real> x(N1, Real(0.0));

        // Boundary conditions are already set

        // Here we consider only the even indices in 2nd direction
        // where normal components are considered
        for (Dim inedx_1 = 0; inedx_1 < N2; ++ ++inedx_1)
        {
            for (Dim inedx_2 = 0; inedx_2 < N3; ++inedx_2)
            {

                Dim stride = stride_func(inedx_1, inedx_2);

                for (Dim index_0 = 1; index_0 < N1 - 1; ++index_0)
                {
                    d[index_0] = rhs.get(stride + index_0);

                    Real gamma_val = -gamma_field.get(stride + index_0);

                    a[index_0] = gamma_val / (dN1 * dN1);
                    b[index_0] = 1.0f - (2.0f * gamma_val) / (dN1 * dN1);
                    c[index_0] = gamma_val / (dN1 * dN1);
                }

                // Solve the tridiagonal system
                thomas_algorithm(a, b, c, d, x);

                // Store the solution
                for (Dim index_0 = 0; index_0 < N1; ++index_0)
                {
                    solution.set(stride + index_0) = x[index_0];
                }
            }
        }

        // Here we consider only the odd indices in 2nd direction
        // where tangential components are considered
        for (Dim inedx_1 = 1; inedx_1 < N2; ++ ++inedx_1)
        {
            for (Dim inedx_2 = 0; inedx_2 < N3; ++inedx_2)
            {

                Dim stride = stride_func(inedx_1, inedx_2);

                a[N1 - 1] = -gamma_field.get(stride + N1 - 1) / (dN1 * dN1);
                b[N1 - 1] = 1.0f + (3.0f * gamma_field.get(stride + N1 - 1)) / (dN1 * dN1);

                for (Dim index_0 = 1; index_0 < N1 - 1; ++index_0)
                {
                    d[index_0] = rhs.get(stride + index_0);
                }

                // Solve the tridiagonal system
                thomas_algorithm(a, b, c, d, x);

                // Store the solution
                for (Dim index_0 = 0; index_0 < N1; ++index_0)
                {
                    solution.set(stride + index_0) = x[index_0];
                }
            }
        }
    };

    // Helpers DIRICHLET BC
    inline float x_at(Dim i) { return i * dx; }
    inline float y_at(Dim j) { return j * dy; }
    inline float z_at(Dim k) { return k * dz; }
    inline float x_half(Dim i) { return (i + 0.5f) * dx; }
    inline float y_half(Dim j) { return (j + 0.5f) * dy; }
    inline float z_half(Dim k) { return (k + 0.5f) * dz; }
    static Real BC_u(Real, Real, Real) { return 1.0f; }
    static Real BC_v(Real, Real, Real) { return 1.0f; }
    static Real BC_w(Real, Real, Real) { return 1.0f; }
};
#endif // SOLVER_HPP