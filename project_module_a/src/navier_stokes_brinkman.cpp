#include "navier_stokes_brinkman.hpp"
#include <stdexcept>

#include <cmath> // for std::abs
#include <fstream>
#include <sstream>

Real NavierStokesBrinkmann::compute_beta(Dim i, Dim j, Dim k) const
{
    Real k_val = k_field.get(i, j, k);
    if (std::fabs(k_val) < 1e-12f)
        k_val = 1e-12f;

    Real nu_val = nu;
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

    Real nu_val = nu;
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

void NavierStokesBrinkmann::initialize_k_field()
{
    for (Dim idx = 0; idx < Nx * Ny * Nz; ++idx)
    {
        Dim i = idx % Nx;
        Dim j = (idx / Nx) % Ny;
        Dim k = idx / (Nx * Ny);

        // Convert grid indices to physical coordinates
        Real x = i * dx;
        Real y = j * dy;
        Real z = k * dz;

        k_field.set(idx) = k_function(x, y, z);
    }
}
void NavierStokesBrinkmann::compute_vector_g(Real t)
{
    // -------------------------------------------------------------------------
    // Purpose:
    //   Assemble the right-hand side vector (g) for the momentum equation:
    //
    //   g = f - ∇p + (ν/2)(Dxx*η0 + Dyy*ζ0 + Dzz*u0) - (ν / (2k)) * u0
    // -------------------------------------------------------------------------

    gradient_pressure_predictor.set(0) = pressure_predictor.getGradient_x();
    gradient_pressure_predictor.set(1) = pressure_predictor.getGradient_y();
    gradient_pressure_predictor.set(2) = pressure_predictor.getGradient_z();

    // Define a target index for debugging prints to avoid console flood
    constexpr Dim DEBUG_I = 1;
    constexpr Dim DEBUG_J = 1;
    constexpr Dim DEBUG_K = 1;
    constexpr Dim DEBUG_COMP = 0; // Check the x-component

    for (Dim comp = 0; comp < vector_rhs.size(); ++comp)
    {
        for (Dim idx = 0; idx < vector_rhs.elements_per_component(); ++idx)
        {
            // Convert linear index to 3D coordinates
            Dim i = idx % Nx;
            Dim j = (idx / Nx) % Ny;
            Dim k = idx / (Nx * Ny);

            // Convert grid indices to physical coordinates
            // this depends on the component, cause of staggered grid!
            Real x = i * dx;
            Real y = j * dy;
            Real z = k * dz;

            // Shift by half a cell in the direction of the component:
            if (comp == 0)
                x += dx / 2.0f;
            else if (comp == 1)
                y += dy / 2.0f;
            else
                z += dz / 2.0f;

            // -----------------------------------------------------------------
            // Directional Laplacian terms
            // -----------------------------------------------------------------
            Real dxx_eta = eta.second_derivative(comp, 0, idx);             // ∂²η/∂x²
            Real dyy_zeta = zeta.second_derivative(comp, 1, idx);           // ∂²ζ/∂y²
            Real dzz_u = velocity_solution.second_derivative(comp, 2, idx); // ∂²u/∂z²
            Real laplacian = dxx_eta + dyy_zeta + dzz_u;

            // -----------------------------------------------------------------
            // Physical properties
            // -----------------------------------------------------------------
            Real nu_val = nu;
            Real k_val = std::max(k_field.get(idx), 1e-12f); // local permeability k

            // Evaluate forcing function
            std::vector<Real> forcing_vec = forcing_function(x, y, z, t);
            Real forcing = forcing_vec[comp];

            Real p_grad = gradient_pressure_predictor.value(comp, idx);

            Real velocity = u_0.value(comp, idx);

            // =================================================================
            // DEBUGGING OUTPUT
            // =================================================================
            if (i == DEBUG_I && j == DEBUG_J && k == DEBUG_K && comp == DEBUG_COMP)
            {
                std::cout << "\n--- DEBUG: Time=" << t << ", Index (" << i << "," << j << "," << k << "), Comp=" << comp << " ---\n";
                std::cout << "k_val (permeability): " << k_val << "\n";
                std::cout << "Laplacian components:\n";
                std::cout << "  Dxx_eta: " << dxx_eta << "\n";
                std::cout << "  Dyy_zeta: " << dyy_zeta << "\n";
                std::cout << "  Dzz_u: " << dzz_u << "\n";
                std::cout << "Laplacian sum: " << laplacian << "\n";
                std::cout << "p_grad: " << p_grad << "\n";
                std::cout << "Forcing: " << forcing << "\n";
            }
            // =================================================================

            // -----------------------------------------------------------------
            // Assemble RHS term
            // -----------------------------------------------------------------
            Real g_val =
                forcing                                      // f
                - p_grad                                     // -∇p
                + Real(0.5) * nu_val * laplacian             // + (ν/2)(∇²η + ∇²ζ + ∇²u)
                - (nu_val / (Real(2.0) * k_val)) * velocity; // - (ν/(2k))u₀

            // =================================================================
            // DEBUGGING OUTPUT - Final result
            // =================================================================
            if (i == DEBUG_I && j == DEBUG_J && k == DEBUG_K && comp == DEBUG_COMP)
            {
                std::cout << "g_val (FINAL): " << g_val << "\n";
                std::cout << "--------------------------------------------------\n";
            }
            // =================================================================

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

void NavierStokesBrinkmann::compute_rhs_pressure()
{
    for (Dim x = 0; x < Nx; ++x)
    {
        for (Dim y = 0; y < Ny; ++y)
        {
            for (Dim z = 0; z < Nz; ++z)
            {
                rhs.set(x, y, z) =
                    -(1.0f / dt) *
                    velocity_solution.divergence(x, y, z);
            }
        }
    }
}

void NavierStokesBrinkmann::solve()
{
    auto stride_x = [this](Dim j, Dim k)
    {
        return j * Nx + k * Nx * Ny;
    };

    auto stride_y = [this](Dim i, Dim k)
    {
        return i + k * Nx * Ny;
    };

    auto stride_z = [this](Dim i, Dim j)
    {
        return i + j * Nx;
    };

    DimensionsHandlerScalar<decltype(stride_x)> x_scalar_handler(Nx, Ny, Nz, dx, stride_x);
    DimensionsHandlerScalar<decltype(stride_y)> y_scalar_handler(Nx, Ny, Nz, dy, stride_y);
    DimensionsHandlerScalar<decltype(stride_z)> z_scalar_handler(Nx, Ny, Nz, dz, stride_z);

    DimensionsHandlerVector<decltype(stride_x)> x_vector_handler(Nx, Ny, Nz, 0, 1, 2, dx, stride_x);
    DimensionsHandlerVector<decltype(stride_y)> y_vector_handler(Nx, Ny, Nz, 1, 0, 2, dy, stride_y);
    DimensionsHandlerVector<decltype(stride_z)> z_vector_handler(Nx, Ny, Nz, 2, 0, 1, dz, stride_z);

    // Initialize solutions
    velocity_solution = u_0;
    pressure_solution = p_0;

    // Time stepping loop
    for (Real t = 0.0f; t < T; t += dt)
    {
        pressure_predictor = pressure_solution;

        // Compute source terms using u_0 (velocity at t^n)
        compute_vector_g(t);
        compute_vector_xi();

        // ============================================================================
        // ===========================MOMENTUM EQUATION SOLVE==========================
        // ============================================================================

        // Solve X-sweep: (I - gamma*Dxx)(eta^{n+1} - eta^n) = xi^{n+1} - eta^n
        vector_rhs = xi - eta;
        velocity_solver.solve<decltype(stride_x), 0>(vector_rhs, vector_intermediate_solution, x_vector_handler);
        eta += vector_intermediate_solution; // Update eta to n+1

        // Solve Y-sweep: (I - gamma*Dyy)(zeta^{n+1} - zeta^n) = eta^{n+1} - zeta^n
        vector_rhs = eta - zeta;
        velocity_solver.solve<decltype(stride_y), 1>(vector_rhs, vector_intermediate_solution, y_vector_handler);
        zeta += vector_intermediate_solution; // Update zeta to n+1

        // Solve Z-sweep: (I - gamma*Dzz)(u^{n+1} - u^n) = zeta^{n+1} - u^n
        // Note: velocity_solution here holds u^n (from initialization or previous loop)
        vector_rhs = zeta - velocity_solution;
        velocity_solver.solve<decltype(stride_z), 2>(vector_rhs, vector_intermediate_solution, z_vector_handler);
        velocity_solution += vector_intermediate_solution; // Update velocity_solution to n+1

        // ============================================================================
        // ===========================PRESSURE EQUATION SOLVE==========================
        // ============================================================================
        compute_rhs_pressure();

        pressure_solver.solve_pressure<decltype(stride_x), 0>(rhs, psi, x_scalar_handler);
        pressure_solver.solve_pressure<decltype(stride_y), 1>(psi, phi, y_scalar_handler);
        pressure_solver.solve_pressure<decltype(stride_z), 2>(phi, other_phi, z_scalar_handler);

        // ============================================================================
        // =====================UPDATE PRESSURE====================
        // ============================================================================
        pressure_solution += other_phi;

        velocity_solver.advance_time();
        pressure_solver.advance_time();

        // -------------------------------------------------------
        // CRITICAL UPDATE: Advance u_0 to the next time step
        // -------------------------------------------------------
        // u_0 must hold the velocity at time 't' for the NEXT iteration's
        // compute_vector_xi() calculation.
        u_0 = velocity_solution;
    }
};