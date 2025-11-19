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

    for (Dim comp = 0; comp < vector_rhs.size(); ++comp)
    {
        for (Dim idx = 0; idx < vector_rhs.elements_per_component(); ++idx)
        {
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
            Real nu_val = nu;                                // local kinematic viscosity ν
            Real k_val = std::max(k_field.get(idx), 1e-12f); // local permeability k //TODO:SET WHEN READING IS BETTER

            // -----------------------------------------------------------------
            // Forcing and pressure gradient
            // -----------------------------------------------------------------
            // Convert linear index to 3D coordinates

            Dim i = idx % Nx;
            Dim j = (idx / Nx) % Ny;
            Dim k = idx / (Nx * Ny);

            // Convert grid indices to physical coordinates
            Real x = i * dx;
            Real y = j * dy;
            Real z = k * dz;

            // Evaluate forcing function
            std::vector<Real> forcing_vec = forcing_function(x, y, z, t);
            Real forcing = forcing_vec[comp];

            // -----------------------------------------------------------------
            // Assemble RHS term
            // -----------------------------------------------------------------

            Real p_grad = gradient_pressure_predictor.value(comp, idx);

            Real velocity = u_0.value(comp, idx);

            Real g_val =
                forcing                                      // f
                - p_grad                                     // -∇p
                + Real(0.5) * nu_val * laplacian             // + (ν/2)(∇²η + ∇²ζ + ∇²u)
                - (nu_val / (Real(2.0) * k_val)) * velocity; // - (ν/(2k))u₀

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
    ScalarVariable div_u(Nx, Ny, Nz, dx, dy, dz);

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

    velocity_solution = u_0;
    pressure_solution = p_0;
    // output method

    for (Real t = 0.0f; t < T; t += dt)
    {
        printf("Time step at t = %.4f\n", t);
        pressure_predictor = pressure_solution + other_phi;
        // std::cout << "Pressure predictor computed.\n";
        compute_vector_g(t);
        // std::cout << "Vector g computed.\n";
        compute_vector_xi();
        // std::cout << "Vector xi computed.\n";

        // ============================================================================
        // ===========================MOMENTUM EQUATION SOLVE==========================
        // ============================================================================
        vector_rhs = xi - eta;
        // std::cout << "Vector RHS for x-direction computed.\n";
        velocity_solver.solve<decltype(stride_x), 0>(vector_rhs, vector_intermediate_solution, x_vector_handler);
        // std::cout << "Velocity intermediate solution for x-direction computed.\n";
        eta += vector_intermediate_solution;
        // std::cout << "Eta updated after x-direction solve.\n";

        vector_rhs = eta - zeta;
        // std::cout << "Vector RHS for y-direction computed.\n";
        velocity_solver.solve<decltype(stride_y), 1>(vector_rhs, vector_intermediate_solution, y_vector_handler);
        // std::cout << "Velocity intermediate solution for y-direction computed.\n";
        zeta += vector_intermediate_solution;
        // std::cout << "Zeta updated after y-direction solve.\n";

        vector_rhs = zeta - velocity_solution;
        // std::cout << "Vector RHS for z-direction computed.\n";
        velocity_solver.solve<decltype(stride_z), 2>(vector_rhs, vector_intermediate_solution, z_vector_handler);
        // std::cout << "Velocity intermediate solution for z-direction computed.\n";
        velocity_solution += vector_intermediate_solution;
        // std::cout << "Velocity solution updated after z-direction solve.\n";

        // ============================================================================
        // ===========================PRESSURE EQUATION SOLVE==========================
        // ============================================================================
        compute_rhs_pressure();
        // std::cout << "RHS for pressure equation computed.\n";
        pressure_solver.solve_pressure<decltype(stride_x), 0>(rhs, psi, x_scalar_handler);
        // std::cout << "Pressure intermediate solution for x-direction computed.\n";
        pressure_solver.solve_pressure<decltype(stride_y), 1>(psi, phi, y_scalar_handler);
        // std::cout << "Pressure intermediate solution for y-direction computed.\n";
        pressure_solver.solve_pressure<decltype(stride_z), 2>(phi, other_phi, z_scalar_handler);
        // std::cout << "Pressure intermediate solution for z-direction computed.\n";

        // ============================================================================
        // =====================UPDATE PRESSURE====================
        // ============================================================================
        pressure_solution += other_phi;
        // std::cout << "Pressure solution updated.\n";
        velocity_solver.advance_time();
        // std::cout << "Velocity solver advanced to next time step.\n";
        pressure_solver.advance_time();
        // std::cout << "Pressure solver advanced to next time step.\n";
    }
};