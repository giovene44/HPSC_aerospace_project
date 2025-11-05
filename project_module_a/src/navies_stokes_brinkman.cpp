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
    ScalarVariables div_u(Nx, Ny, Nz, dx, dy, dz);

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

void NavierStokesBrinkmann::block_solver(int derivation_direction, const ScalarVariables &rhs, const ScalarVariables &gamma, ScalarVariables &solution)
{
    if (derivation_direction < 0 || derivation_direction > 2)
        throw std::invalid_argument("Invalid derivation direction. Must be 0, 1, or 2.");

    // this method decomposes the 3D problem into multiple 1D tridiagonal systems.
    // this decomposition is based on the specified direction (0 for x, 1 for y, 2 for z).

    if (derivation_direction == 0)
    {
        for (Dim j = 0; j < Ny; ++j)
        {
            for (Dim k = 0; k < Nz; ++k)
            {

                Dim stride = j * Nx + k * Nx * Ny;

                std::vector<float> a(Nx - 1, 0.0f);
                std::vector<float> b(Nx, 0.0f);
                std::vector<float> c(Nx - 1, 0.0f);
                std::vector<float> d(Nx, 0.0f);
                std::vector<float> x(Nx, 0.0f);

                std::vector<float> gamma_line(Nx, 0.0f);
                for (Dim i = 0; i < Nx; ++i)
                {
                    gamma_line[i] = -gamma.get(stride + i);
                }

                // a = gamma line without first element
                for (Dim i = 1; i < Nx; ++i)
                {
                    a[i - 1] = gamma_line[i];
                }

                // b = 1 + 2*gamma line
                for (Dim i = 0; i < Nx; ++i)
                {
                    b[i] = 1.0f - 2.0f * gamma_line[i];
                }

                // c = gamma line without last element
                for (Dim i = 0; i < Nx - 1; ++i)
                {
                    c[i] = gamma_line[i + 1];
                }

                // d = rhs
                for (Dim i = 0; i < Nx; ++i)
                {
                    d[i] = rhs.get(stride + i);
                }

                // Solve the tridiagonal system
                thomas_algorithm(a, b, c, d, x);

                // Store the solution
                for (Dim i = 0; i < Nx; ++i)
                {
                    solution.set(stride + i) = x[i];
                }
            }
        }
    }

    if (derivation_direction == 1)
    {
        for (Dim i = 0; i < Nx; ++i)
        {
            for (Dim k = 0; k < Nz; ++k)
            {

                Dim stride = i + k * Nx * Ny;

                std::vector<float> a(Ny - 1, 0.0f);
                std::vector<float> b(Ny, 0.0f);
                std::vector<float> c(Ny - 1, 0.0f);
                std::vector<float> d(Ny, 0.0f);
                std::vector<float> x(Ny, 0.0f);

                std::vector<float> gamma_line(Ny, 0.0f);
                for (Dim j = 0; j < Ny; ++j)
                {
                    gamma_line[j] = -gamma.get(stride + j * Nx);
                }

                // a = gamma line without first element
                for (Dim j = 1; j < Ny; ++j)
                {
                    a[j - 1] = gamma_line[j];
                }
                // b = 1 + 2*gamma line
                for (Dim j = 0; j < Ny; ++j)
                {
                    b[j] = 1.0f - 2.0f * gamma_line[j];
                }
                // c = gamma line without last element
                for (Dim j = 0; j < Ny - 1; ++j)
                {
                    c[j] = gamma_line[j + 1];
                }
                // d = rhs
                for (Dim j = 0; j < Ny; ++j)
                {
                    d[j] = rhs.get(stride + j * Nx);
                }
                // Solve the tridiagonal system
                thomas_algorithm(a, b, c, d, x);
                // Store the solution
                for (Dim j = 0; j < Ny; ++j)
                {
                    solution.set(stride + j * Nx) = x[j];
                }
            }
        }
    }

    if (derivation_direction == 2)
    {
        for (Dim i = 0; i < Nx; ++i)
        {
            for (Dim j = 0; j < Ny; ++j)
            {

                Dim stride = i + j * Nx;

                std::vector<float> a(Nz - 1, 0.0f);
                std::vector<float> b(Nz, 0.0f);
                std::vector<float> c(Nz - 1, 0.0f);
                std::vector<float> d(Nz, 0.0f);
                std::vector<float> x(Nz, 0.0f);

                std::vector<float> gamma_line(Nz, 0.0f);
                for (Dim k = 0; k < Nz; ++k)
                {
                    gamma_line[k] = -gamma.get(stride + k * Nx * Ny);
                }

                // a = gamma line without first element
                for (Dim k = 1; k < Nz; ++k)
                {
                    a[k - 1] = gamma_line[k];
                }
                // b = 1 + 2*gamma line
                for (Dim k = 0; k < Nz; ++k)
                {
                    b[k] = 1.0f - 2.0f * gamma_line[k];
                }
                // c = gamma line without last element
                for (Dim k = 0; k < Nz - 1; ++k)
                {
                    c[k] = gamma_line[k + 1];
                }
                // d = rhs
                for (Dim k = 0; k < Nz; ++k)
                {
                    d[k] = rhs.get(stride + k * Nx * Ny);
                }
                // Solve the tridiagonal system
                thomas_algorithm(a, b, c, d, x);
                // Store the solution
                for (Dim k = 0; k < Nz; ++k)
                {
                    solution.set(stride + k * Nx * Ny) = x[k];
                }
            }
        }
    }
}

template <typename StrideFunc>
void NavierStokesBrinkmann::block_solver(const ScalarVariables &rhs, ScalarVariables &solution, const DimensionsHandler<StrideFunc> &dim_hand)
{
    // ----------------------------------------------------------------------------
    // Implementation of the block_solver method for the pressure: (no gamma needed here)
    // it is still a naive implementation that doesn't take into account boundary conditions
    // i redefine a,b,c,d every time because the thomas_algorithm does modify those vectors!,
    // moreover when we will add the BCs we will need to modify a,b,c in specific positions,
    // hence in specific iterations of the loops

    // this method decomposes the 3D problem into multiple 1D tridiagonal systems.
    // this decomposition is based on the specified direction (0 for x, 1 for y, 2 for z).
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