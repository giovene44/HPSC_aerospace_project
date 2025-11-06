#include "navier_stokes_brinkman.hpp"
#include <stdexcept>

#include <cmath> // for std::abs

Real NavierStokesBrinkmann::compute_beta(Dim i, Dim j, Dim k) const
{
    Real k_val = k_field.get(i, j, k);
    if (std::fabs(k_val) < 1e-12f)
        k_val = 1e-12f;

    Real nu_val = nu.get(i, j, k);
    return 1.0f + (dt * nu_val) / (2.0f * k_val);
}

Real NavierStokesBrinkmann::compute_beta(Dim index) const
{
    Dim i = index % Nx;
    Dim j = (index / Nx) % Ny;
    Dim k = index / (Nx * Ny);
    return compute_beta(i, j, k);
}

Real NavierStokesBrinkmann::compute_gamma(Dim i, Dim j, Dim k) const
{
    Real k_val = k_field.get(i, j, k);
    if (std::fabs(k_val) < 1e-12f)
        k_val = 1e-12f;

    Real nu_val = nu.get(i, j, k);
    Real beta = 1.0f + (dt * nu_val) / (2.0f * k_val);
    return (dt * nu_val) / (2.0f * beta);
}

Real NavierStokesBrinkmann::compute_gamma(Dim index) const
{
    Dim i = index % Nx;
    Dim j = (index / Nx) % Ny;
    Dim k = index / (Nx * Ny);
    return compute_gamma(i, j, k);
}

void NavierStokesBrinkmann::initialize_gamma_field()
{
    for (Dim idx = 0; idx < Nx * Ny * Nz; ++idx)
    {
        Dim i = idx % Nx;
        Dim j = (idx / Nx) % Ny;
        Dim k = idx / (Nx * Ny);
        gamma_field.set(idx) = compute_gamma(i, j, k);
    }
}

void NavierStokesBrinkmann::compute_vector_difference(VectorVariable &output, const VectorVariable &v1, const VectorVariable &v2)
{
    // output must be preallocated and have matching dimensions
    for (int a = 0; a < output.size(); ++a)
    {
        for (int i = 0; i < Nx; ++i)
        {
            for (int j = 0; j < Ny; ++j)
            {
                for (int k = 0; k < Nz; ++k)
                {
                    output.set(a, i, j, k) =
                        v1.value(a, i, j, k) - v2.value(a, i, j, k);
                }
            }
        }
    }
}

void NavierStokesBrinkmann::compute_gradient_pressure_field()
{
    for (Dim comp = 0; comp < gradient_pressure.size(); ++comp)
    {
        for (Dim idx = 0; idx < gradient_pressure.elements_per_component(); ++idx)
        {
            Real grad = 0.0f;
            if (comp == 0)
                grad = p_0.getGradient_x(idx);
            else if (comp == 1)
                grad = p_0.getGradient_y(idx);
            else if (comp == 2)
                grad = p_0.getGradient_z(idx);
            gradient_pressure.set(comp, idx) = grad;
        }
    }
}

void NavierStokesBrinkmann::compute_vector_g()
{
    // -------------------------------------------------------------------------
    // Purpose:
    //   Assemble the right-hand side vector (g) for the momentum equation:
    //
    //   g = f - ∇p + (ν/2)(Dxx*η0 + Dyy*ζ0 + Dzz*u0) - (ν / (2k)) * u0
    // -------------------------------------------------------------------------

    for (int comp = 0; comp < vector_rhs.size(); ++comp)
    {
        for (Dim idx = 0; idx < vector_rhs.elements_per_component(); ++idx)
        {
            // -----------------------------------------------------------------
            // Directional Laplacian terms
            // -----------------------------------------------------------------
            float dxx_eta = eta_0.second_derivative(comp, 0, idx);   // ∂²η/∂x²
            float dyy_zeta = zeta_0.second_derivative(comp, 1, idx); // ∂²ζ/∂y²
            float dzz_u = u_0.second_derivative(comp, 2, idx);       // ∂²u/∂z²

            // -----------------------------------------------------------------
            // Physical properties
            // -----------------------------------------------------------------
            float nu_val = nu.get(idx);     // local kinematic viscosity ν
            float k_val = k_field.get(idx); // local permeability k
            if (std::abs(k_val) < 1e-12f)
                k_val = 1e-12f; // avoid division by zero

            // -----------------------------------------------------------------
            // Forcing and pressure gradient
            // -----------------------------------------------------------------
            float forcing = f.value(comp, idx); // external forcing term
            compute_gradient_pressure_field();
            float grad_p = gradient_pressure.value(comp, idx);

            // -----------------------------------------------------------------
            // Assemble RHS term
            // -----------------------------------------------------------------

            float g_val =
                forcing                                             // f
                - grad_p                                            // -∇p
                + 0.5f * nu_val * (dxx_eta + dyy_zeta + dzz_u)      // + (ν/2)(∇²η + ∇²ζ + ∇²u)
                - (nu_val / (2.0f * k_val)) * u_0.value(comp, idx); // - (ν/(2k))u₀

            // -----------------------------------------------------------------
            // Store result
            // -----------------------------------------------------------------
            g.set(comp, idx) = g_val;
        }
    }
}

void NavierStokesBrinkmann::compute_vector_xi()
{
    for (Dim comp = 0; comp < xi.size(); ++comp)
    {
        for (Dim idx = 0; idx < xi.elements_per_component(); ++idx)
        {
            xi.set(comp, idx) =
                u_0.value(comp, idx) + (dt / compute_beta(idx)) * g.value(comp, idx);
        }
    }
}

void NavierStokesBrinkmann::compute_vector_gamma_D_term(int direction)
{
    for (Dim comp = 0; comp < 3; ++comp)
    {
        for (Dim idx = 0; idx < vector_gamma_D_term.elements_per_component(); ++idx)
        {
            Real D_term = 0.0f;
            if (direction == 0)
            {
                D_term = eta_0.second_derivative(comp, 0, idx); // ∂xx η
            }
            else if (direction == 1)
            {
                D_term = zeta_0.second_derivative(comp, 1, idx); // ∂yy ζ
            }
            else if (direction == 2)
            {
                D_term = u_0.second_derivative(comp, 2, idx); // ∂zz u
            }
            Real gamma_val = gamma_field.get(idx);
            vector_gamma_D_term.set(comp, idx) = gamma_val * D_term;
        }
    }
}

// ----------------------------------------------------------------------------
// Purpose:
//   Compute the right-hand side vector for the momentum equation by
//   calculating the difference between two input vector fields.
// ----------------------------------------------------------------------------
void NavierStokesBrinkmann::compute_vector_rhs(const VectorVariable &vector1, const VectorVariable &vector2)
{
    compute_vector_difference(vector_rhs, vector1, vector2);
}

void NavierStokesBrinkmann::compute_scalar_rhs_pressure_1()
{
    ScalarVariable div_u(Nx, Ny, Nz, dx, dy, dz);

    for (int x = 0; x < Nx; ++x)
    {
        for (int y = 0; y < Ny; ++y)
        {
            for (int z = 0; z < Nz; ++z)
            {
                rhs.set(x, y, z) =
                    -(1.0f / dt) *
                    u_1.divergence(x, y, z);
            }
        }
    }
}

void NavierStokesBrinkmann::impose_Neumann_bc_scalar(int direction, int side, float neumann_boundary_value)
{
    float value, c;
    if (direction == 0)
    {

        int i;

        if (side == 0)
        {
            i = 0;
            value = 2.0f * dx * neumann_boundary_value;
        }

        else
        {
            i = Nx - 1;
            c = (1 + 1.0f / dx * dx);
            value = -dx * c * neumann_boundary_value;
        }

        for (int y = 0; y < Ny; y++)
        {
            for (int z = 0; z < Nz; z++)
            {
                rhs.set(i, y, z) = rhs.get(i, y, z) + value;
            }
        }
    }
    if (direction == 1)
    {
        int y;
        if (side == 0)
        {
            y = 0;
            value = 2.0f * dy * neumann_boundary_value;
        }

        else
        {
            y = Ny - 1;
            c = (1 + 1.0f / dy * dy);
            value = -dy * c * neumann_boundary_value;
        }

        for (int i = 0; i < Nx; i++)
        {
            for (int z = 0; z < Nz; z++)
            {
                rhs.set(i, y, z) = rhs.get(i, y, z) + value;
            }
        }
    }
    if (direction == 2)
    {
        int z;
        if (side == 0)
        {
            z = 0;
            value = 2.0f * dz * neumann_boundary_value;
        }

        else
        {
            z = Nz - 1;
            c = (1 + 1.0f / dz * dz);
            value = -dz * c * neumann_boundary_value;
        }

        for (int i = 0; i < Nx; i++)
        {
            for (int y = 0; y < Ny; y++)
            {
                rhs.set(i, y, z) = rhs.get(i, y, z) + value;
            }
        }
    }
}

template <typename StrideFunc>
void NavierStokesBrinkmann::block_solver(const ScalarVariable &rhs, const ScalarVariable &gamma, ScalarVariable &solution, const DimensionsHandler<StrideFunc> &dim_hand)
{
    using N1 = dim_hand.N1;
    using N2 = dim_hand.N2;
    using N3 = dim_hand.N3;
    using dN1 = dim_hand.dN1;
    using stride_func = dim_hand.stride;

    std::vector<Real> a(N1, Real(0.0));
    std::vector<Real> b(N1, Real(1.0));
    std::vector<Real> c(N1, Real(0.0));
    std::vector<Real> d(N1, Real(0.0));
    std::vector<Real> x(N1, Real(0.0));

    // Boundary conditions are already set

    // Here we consider only the even indices in 2nd direction
    // where normal components are considered
    for (Dim inedx_1 = 0; inedx_1 < N2; ++++inedx_1)
    {
        for (Dim inedx_2 = 0; inedx_2 < N3; ++inedx_2)
        {

            Dim stride = stride_func(inedx_1, inedx_2);


            for (Dim index_0 = 1; index_0 < N1 - 1; ++index_0)
            {
                d[index_0] = rhs.get(stride + index_0);

                Real gamma_val = -gamma.get(stride + index_0);

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
    for (Dim inedx_1 = 1; inedx_1 < N2; ++++inedx_1)
    {
        for (Dim inedx_2 = 0; inedx_2 < N3; ++inedx_2)
        {

            Dim stride = stride_func(inedx_1, inedx_2);

            a[N1 - 1] =  -gamma.get(stride + N1 - 1) / (dN1 * dN1);
            b[N1 - 1] = 1.0f + (3.0f * gamma.get(stride + N1 - 1)) / (dN1 * dN1);

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
}

template <typename StrideFunc>
void NavierStokesBrinkmann::block_solver(const ScalarVariable &rhs, ScalarVariable &solution, const DimensionsHandler<StrideFunc> &dim_hand)
{
    // ----------------------------------------------------------------------------
    // Implementation of the block_solver method for the pressure.
    // this method decomposes the 3D problem into multiple 1D tridiagonal systems.

    using N1 = dim_hand.N1;
    using N2 = dim_hand.N2;
    using N3 = dim_hand.N3;
    using dN1 = dim_hand.dN1;
    using stride_func = dim_hand.stride;

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
}