#include "navier_stokes_brinkman.hpp"
#include <stdexcept>

#include <cmath> // for std::abs
#include <fstream>
#include <sstream>
#include <helper.hpp>

Real NavierStokesBrinkmann::compute_beta(Dim i, Dim j, Dim k) const
{
    Real k_val = k_field.get(i, j, k);
    if (std::fabs(k_val) < 1e-12f)
        k_val = 1e-12f;

    return 1.0f + (dt * nu) / (2.0f * k_val);
}

Real NavierStokesBrinkmann::compute_beta(Dim index) const
{
    Dim i = index % Nx;
    Dim j = (index / Nx) % Ny;
    Dim k = index / (Nx * Ny);
    return compute_beta(i, j, k);
}

void NavierStokesBrinkmann::center_pressure(ScalarVariable &pressure_field)
{
    Real average = 0.0f;
    Real total_elements = 0.0f;

    for (Dim idx = 0; idx < Nx; ++idx)
    {
        for(Dim idy = 0; idy< Ny; ++idy)
        {
            for(Dim idz = 0; idz < Nz; ++idz)
            {
                Dim weight = 1;
                if (idx == 0 || idx == Nx - 1) weight *= 0.5;
                if (idy == 0 || idy == Ny - 1) weight *= 0.5;
                if (idz == 0 || idz == Nz - 1) weight *= 0.5;
                average += pressure_field.get(idx, idy, idz) * weight;

                total_elements += weight;

            }
        }
    }


    average /= total_elements;

    for (Dim idx = 0; idx < Nx * Ny * Nz; ++idx)
    {
        pressure_field.set(idx) = pressure_field.get(idx) - average;
    }
}

Real NavierStokesBrinkmann::compute_gamma(Dim i, Dim j, Dim k) const
{
    Real k_val = k_field.get(i, j, k);
    if (std::fabs(k_val) < 1e-12f)
        k_val = 1e-12f;

    Real beta = 1.0f + (dt * nu) / (2.0f * k_val);
    return (dt * nu) / (2.0f * beta);
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
            Real velocity = velocity_solution.value(comp, idx);

            // =================================================================
            // DEBUGGING OUTPUT
            // =================================================================
            /*
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


            */

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
            /*
              if (i == DEBUG_I && j == DEBUG_J && k == DEBUG_K && comp == DEBUG_COMP)
            {
                std::cout << "g_val (FINAL): " << g_val << "\n";
                std::cout << "--------------------------------------------------\n";
            }

            */

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
                velocity_solution.value(comp, idx) + (dt / compute_beta(idx)) * g.value(comp, idx);
        }
    }
}

void NavierStokesBrinkmann::compute_rhs_pressure()
{
    for (Dim x = 1; x < Nx; ++x)
    {
        for (Dim y = 1; y < Ny; ++y)
        {
            for (Dim z = 1; z < Nz; ++z)
            {
                rhs.set(x, y, z) =
                    -(1.0f / dt) *
                    velocity_solution.divergence(x, y, z);
            }
        }
    }


    //We Impose div(u)=0 at boundaries:

    for (Dim j = 1; j < Ny; ++j)
    {
        for (Dim k = 1; k < Nz; ++k)
        { 
            rhs.set(0, j, k) = 0.0;
        }
    }

    for (Dim i = 1; i < Nx; ++i)
    {
        for (Dim k = 1; k < Nz; ++k)
        {
            rhs.set(i, 0, k) = 0.0;
        }
    }

    for (Dim i = 1; i < Nx; ++i)
    {
        for (Dim j = 1; j < Ny; ++j)
        { 
            rhs.set(i, j, 0) = 0.0;
        }
    }


    rhs.set(0, 0, 0) = 0.0;

    // edge (x,0,0):
    for(Dim i = 1; i < Nx; ++i)
    {
        rhs.set(i, 0, 0) = 0.0;
    }

    // edge (0,y,0):
    for(Dim j = 1; j < Ny; ++j)
    {
        rhs.set(0, j, 0) = 0.0;
    }
    // edge (0,0,z):
    for(Dim k = 1; k < Nz; ++k)
    {
        rhs.set(0, 0, k) = 0.0;
    }
}

void NavierStokesBrinkmann::solve(const ManufacturedSolution &mms)
{
    auto stride_x = [this](Dim j, Dim k)
    { return j * Nx + k * Nx * Ny; };
    auto stride_y = [this](Dim i, Dim k)
    { return i + k * Nx * Ny; };
    auto stride_z = [this](Dim i, Dim j)
    { return i + j * Nx; };

    DimensionsHandlerScalar<decltype(stride_x)> x_scalar_handler(Nx, Ny, Nz, dx, stride_x);
    DimensionsHandlerScalar<decltype(stride_y)> y_scalar_handler(Nx, Ny, Nz, dy, stride_y);
    DimensionsHandlerScalar<decltype(stride_z)> z_scalar_handler(Nx, Ny, Nz, dz, stride_z);

    DimensionsHandlerVector<decltype(stride_x)> x_vector_handler(Nx, Ny, Nz, 0, 1, 2, dx, stride_x);
    DimensionsHandlerVector<decltype(stride_y)> y_vector_handler(Nx, Ny, Nz, 1, 0, 2, dy, stride_y);
    DimensionsHandlerVector<decltype(stride_z)> z_vector_handler(Nx, Ny, Nz, 2, 0, 1, dz, stride_z);

    // Initialize
    velocity_solution.set_all(u_boundary, Real(0.0));
    pressure_solution.set_all(p_boundary, Real(0.0));

    // Init intermediate vars to avoid junk values
    xi = eta = zeta = velocity_solution;
    psi = phi = other_phi = pressure_solution;

    velocity_time_series.clear();
    pressure_time_series.clear();
    velocity_time_series.emplace_back(velocity_solution);
    pressure_time_series.emplace_back(pressure_solution);

    // --- Time Stepping Loop ---
    for (Real t = dt; t <= T; t += dt)
    {
        // 1. Momentum Predictor Step (Calculate u*)
        // ------------------------------------------
        compute_vector_g(t);
        compute_vector_xi();

        // X-Sweep
        vector_rhs = xi - eta;
        velocity_solver.solve<decltype(stride_x), 0>(vector_rhs, vector_intermediate_solution, x_vector_handler);
        eta += vector_intermediate_solution;

        // Y-Sweep
        vector_rhs = eta - zeta;
        velocity_solver.solve<decltype(stride_y), 1>(vector_rhs, vector_intermediate_solution, y_vector_handler);
        zeta += vector_intermediate_solution;

        // Z-Sweep
        vector_rhs = zeta - velocity_solution;
        velocity_solver.solve<decltype(stride_z), 2>(vector_rhs, vector_intermediate_solution, z_vector_handler);

        // Update to Intermediate Velocity u*
        velocity_solution += vector_intermediate_solution;

        // 2. Pressure Projection Step (Calculate phi)
        // -------------------------------------------
        compute_rhs_pressure(); // RHS = -div(u*) / dt

        // Solve Poisson Equation: Laplacian(phi) = RHS
        pressure_solver.solve_pressure<decltype(stride_x), 0>(rhs, psi, x_scalar_handler);
        pressure_solver.solve_pressure<decltype(stride_y), 1>(psi, phi, y_scalar_handler);
        pressure_solver.solve_pressure<decltype(stride_z), 2>(phi, other_phi, z_scalar_handler);

        // 3. Update Fields
        // -------------------------------------------

        // Update Pressure: p^{n+1} = phi (assuming phi is total pressure from BCs)
        pressure_solution += other_phi;
        center_pressure(pressure_solution);
        /*
        // Correct Velocity: u^{n+1} = u* - (dt/beta) * grad(phi)
        // This projects velocity onto the divergence-free space
        for (Dim k = 0; k < Nz; ++k)
        {
            for (Dim j = 0; j < Ny; ++j)
            {
                for (Dim i = 0; i < Nx; ++i)
                {
                    Real beta_val = compute_beta(i, j, k);
                    Real coeff = dt / beta_val;

                    // --- Correct U (Component 0) ---
                    // Condition: Interior points only (i=0 and i=Nx-1 are fixed Dirichlet)
                    if (i > 0 && i < Nx - 1)
                    {
                        // Central Difference: (P_{i+1} - P_{i-1}) / 2dx
                        Real dp_dx = (other_phi.get(i + 1, j, k) - other_phi.get(i - 1, j, k)) / (2.0 * dx);
                        velocity_solution.set(0, i, j, k) -= coeff * dp_dx;
                    }

                    // --- Correct V (Component 1) ---
                    // Condition: Interior points only (j=0 and j=Ny-1 are fixed Dirichlet)
                    if (j > 0 && j < Ny - 1)
                    {
                        // Central Difference: (P_{j+1} - P_{j-1}) / 2dy
                        Real dp_dy = (other_phi.get(i, j + 1, k) - other_phi.get(i, j - 1, k)) / (2.0 * dy);
                        velocity_solution.set(1, i, j, k) -= coeff * dp_dy;
                    }

                    // --- Correct W (Component 2) ---
                    // Condition: Interior points only (k=0 and k=Nz-1 are fixed Dirichlet)
                    if (k > 0 && k < Nz - 1)
                    {
                        // Central Difference: (P_{k+1} - P_{k-1}) / 2dz
                        Real dp_dz = (other_phi.get(i, j, k + 1) - other_phi.get(i, j, k - 1)) / (2.0 * dz);
                        velocity_solution.set(2, i, j, k) -= coeff * dp_dz;
                    }
                }
            }
        }

        */

        velocity_time_series.emplace_back(velocity_solution);
        pressure_time_series.emplace_back(pressure_solution);
    }
}