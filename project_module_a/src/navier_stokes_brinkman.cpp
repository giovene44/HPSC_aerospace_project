#include "navier_stokes_brinkman.hpp"
#include <stdexcept>

#include <cmath> // for std::abs
#include <fstream>
#include <sstream>

void NavierStokesBrinkmann::parse_input(const std::string &input_file)
{
    std::ifstream file(input_file);
    if (!file.is_open())
    {
        std::cerr << "Error - Cannot open file " << input_file << std::endl;
    }

    // Lambda function used just to read the input file
    auto next_value = [&](auto &var) {
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            iss >> var;
            return;
        }
    };

    std::string u0_init_file, p0_init_file, k_file;

    // ========= Mesh dimensions ==========
    next_value(Nx);
    next_value(Ny);
    next_value(Nz);

    // ========== Time parameters ==========
    next_value(dt);
    next_value(T);

    // ========== Spatial parameters ==========
    next_value(dx);
    next_value(dy);
    next_value(dz);

    // ========= Initial values ==========
    next_value(u0_init_file);
    next_value(p0_init_file);
    next_value(k_file);
}


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


void NavierStokesBrinkmann::compute_vector_summatory(VectorVariable &output, const VectorVariable &v1, const VectorVariable &v2)
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
                        v1.value(a, i, j, k) + v2.value(a, i, j, k);
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

void NavierStokesBrinkmann::update_pressure_and_velocity_fields()
{
    // Update pressure field
    for (Dim idx = 0; idx < Nx * Ny * Nz; ++idx)
    {
        pressure_solution.set(idx) += other_phi.get(idx);
    }

    // Update velocity field
    compute_gradient_pressure_field();
    for (int comp = 0; comp < velocity_solution.size(); ++comp)
    {
        for (Dim idx = 0; idx < velocity_solution.elements_per_component(); ++idx)
        {
            Real grad_p = gradient_pressure.value(comp, idx);
            velocity_solution.set(comp, idx) =
                u_1.value(comp, idx) - (dt / compute_beta(idx)) * grad_p;
        }
    }
}

void NavierStokesBrinkmann::solve()
{

    for (Real t = 0.0f; t < T; t += dt)
    {
    // ============================================================================
    // MOMENTUM EQUATION SOLVE
    // ============================================================================
        compute_vector_g();
        compute_vector_xi();

        compute_vector_difference(vector_rhs, xi, eta_0);
        u_solver.solve_x_dir(vector_rhs.x(), vector_sol_tmp.x());
        v_solver.solve_x_dir(vector_rhs.y(), vector_sol_tmp.y());
        w_solver.solve_x_dir(vector_rhs.z(), vector_sol_tmp.z());
        compute_vector_summatory(eta_1, vector_sol_tmp, eta_0);

        compute_vector_difference(vector_rhs, eta_1, zeta_0);
        u_solver.solve_y_dir(vector_rhs.x(), vector_sol_tmp.x());
        v_solver.solve_y_dir(vector_rhs.y(), vector_sol_tmp.y());
        w_solver.solve_y_dir(vector_rhs.z(), vector_sol_tmp.z());
        compute_vector_summatory(zeta_1, vector_sol_tmp, zeta_0);

        compute_vector_difference(vector_rhs, zeta_1, u_0);
        u_solver.solve_z_dir(vector_rhs.x(), vector_sol_tmp.x());
        v_solver.solve_z_dir(vector_rhs.y(), vector_sol_tmp.y());
        w_solver.solve_z_dir(vector_rhs.z(), vector_sol_tmp.z());
        compute_vector_summatory(u_1, vector_sol_tmp, u_0);


    // ============================================================================
    // PRESSURE EQUATION SOLVE
    // ============================================================================
        compute_scalar_rhs_pressure_1();
        p_solver.solve_x_dir(rhs, psi);
        p_solver.solve_y_dir(psi, phi);
        p_solver.solve_z_dir(phi, other_phi);



    // ============================================================================
    // UPDATE PRESSURE AND VELOCITY FIELDS
    // ============================================================================
       update_pressure_and_velocity_fields();
    }
};
