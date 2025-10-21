#include "navier_stokes_brinkman.hpp"
#include "VectorVariable.hpp"

#include "navier_stokes_brinkman.hpp"
#include "VectorVariable.hpp"
#include <stdexcept>

#include <cmath> // for std::abs

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

Real NavierStokesBrinkmann::compute_beta(Dim i, Dim j, Dim k) const
{
    Real k_val = k_field.get(i, j, k);
    if (std::fabs(k_val) < 1e-12f)
        k_val = 1e-12f;

    Real nu_val = nu.value(0, i, j, k);
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

    Real nu_val = nu.value(0, i, j, k);
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
            float nu_val = nu.value(comp, idx); // local viscosity ν
            float k_val = k_field.get(idx);     // local permeability k
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
            float rhs_val =
                forcing                                             // f
                - grad_p                                            // -∇p
                + 0.5f * nu_val * (dxx_eta + dyy_zeta + dzz_u)      // + (ν/2)(∇²η + ∇²ζ + ∇²u)
                - (nu_val / (2.0f * k_val)) * u_0.value(comp, idx); // - (ν/(2k))u₀

            // -----------------------------------------------------------------
            // Store result
            // -----------------------------------------------------------------
            vector_rhs.set(comp, idx) = rhs_val;
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
            Real gamma_val = compute_gamma(idx);
            vector_gamma_D_term.set(comp, idx) = gamma_val * D_term;
        }
    }
}

void NavierStokesBrinkmann::compute_vector_rhs(const VectorVariable &vector1, const VectorVariable &vector2)
{
    compute_vector_difference(vector_rhs, vector1, vector2);
}
