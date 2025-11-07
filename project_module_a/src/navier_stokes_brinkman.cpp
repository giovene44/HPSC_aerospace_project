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
            Real dxx_eta = eta.second_derivative(comp, 0, idx);   // ∂²η/∂x²
            Real dyy_zeta = zeta.second_derivative(comp, 1, idx); // ∂²ζ/∂y²
            Real dzz_u = u_0.second_derivative(comp, 2, idx);     // ∂²u/∂z²

            // -----------------------------------------------------------------
            // Physical properties
            // -----------------------------------------------------------------
            Real nu_val = nu.get(idx);     // local kinematic viscosity ν
            Real k_val = k_field.get(idx); // local permeability k
            if (std::abs(k_val) < 1e-12f)
                k_val = 1e-12f; // avoid division by zero

            // -----------------------------------------------------------------
            // Forcing and pressure gradient
            // -----------------------------------------------------------------
            Real forcing = f.value(comp, idx); // external forcing term
            compute_gradient_pressure_field();
            Real grad_p = gradient_pressure.value(comp, idx);

            // -----------------------------------------------------------------
            // Assemble RHS term
            // -----------------------------------------------------------------

            Real g_val =
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
                velocity_solution.value(comp, idx) + (dt / compute_beta(idx)) * g.value(comp, idx);
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
                D_term = eta.second_derivative(comp, 0, idx); // ∂xx η
            }
            else if (direction == 1)
            {
                D_term = zeta.second_derivative(comp, 1, idx); // ∂yy ζ
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

void NavierStokesBrinkmann::compute_scalar_rhs_pressure()
{
    for (Dim i = 0; i < Nx; ++i)
    {
        for (Dim j = 0; j < Ny; ++j)
        {
            for (Dim k = 0; k < Nz; ++k)
            {
                Dim index = i + j * Nx + k * Nx * Ny;

                rhs.set(index) = -(velocity_solution.divergence(index)) / dt;
            }
        }
    }
}

template <typename StrideFunction>
void impose_Dirichlet(VectorVariable &rhs, const DimensionsHandlerVector<StrideFunction> &dim_hand)
{
    using N1 = dim_hand.N1;
    using N2 = dim_hand.N2;
    using N3 = dim_hand.N3;
    using comp1 = dim_hand.comp1;
    using comp2 = dim_hand.comp2;
    using comp3 = dim_hand.comp3;
    using dN1 = dim_hand.dN1;
    using stride_func = dim_hand.stride;

    // Here we consider only the even indices in 2nd direction
    // where normal components are considered
    for (Dim comp = 0; comp < 3; ++comp)
    {
        for (Dim inedx_1 = 0; inedx_1 < N2; ++ ++inedx_1)
        {
            for (Dim inedx_2 = 0; inedx_2 < N3; ++inedx_2)
            {
                Dim stride = stride_func(inedx_1, inedx_2);

                rhs.set(comp, stride + 0) = u_boundary.value(comp, stride + 0) - (u_boundary.first_derivative(comp2, stride + 0) + u_boundary.first_derivative(comp3, stride + 0)) * dN1 * Real(0.5);
                rhs.set(comp, stride + N1 - 1) = u_boundary.value(comp, stride + N1 - 1);
            }
        }

        // Here we consider only the odd indices in 2nd direction
        // where tangential components are considered
        for (Dim inedx_1 = 1; inedx_1 < N2; ++ ++inedx_1)
        {
            for (Dim inedx_2 = 0; inedx_2 < N3; ++inedx_2)
            {

                Dim stride = stride_func(inedx_1, inedx_2);

                rhs.set(comp, stride + 0) = u_boundary.value(comp, stride + 0);
                rhs.set(comp, stride + N1 - 1) = f + Real(2.0) / (dN1 * dN1) * u_boundary.value(comp, stride + N1 - 1);
            }
        }
    }
};

template <typename StrideFunction>
void impose_Neumann(ScalarVariable &rhs, const DimensionsHandlerScalar<StrideFunction> &dim_hand)
{
    using N1 = dim_hand.N1;
    using N2 = dim_hand.N2;
    using N3 = dim_hand.N3;
    using dN1 = dim_hand.dN1;
    using stride_func = dim_hand.stride;

    // Implementation of Neumann boundary condition
    using N1 = dim_hand.N1;
    using N2 = dim_hand.N2;
    using N3 = dim_hand.N3;
    using dN1 = dim_hand.dN1;
    using stride_func = dim_hand.stride;

    for (Dim inedx_1 = 0; inedx_1 < N2; ++inedx_1)
    {
        for (Dim inedx_2 = 0; inedx_2 < N3; ++inedx_2)
        {
            Dim stride = stride_func(inedx_1, inedx_2);

            // Lower boundary (index 0)
            rhs.set(stride + 0) = f - Real(2.0) / dN1 * p_boundary.get(stride + 0); // ∂p/∂n = f at "left" boundary

            // Upper boundary (index N1-1)
            rhs.set(stride + N1 - 1) = f + Real(2.0) / dN1 * p_boundary.get(stride + N1 - 1); // ∂p/∂n = f at "right" boundary
        }
    }
};

// Block solver for velocity components
template <typename StrideFunc>
void NavierStokesBrinkmann::block_solver(const VectorVariable &rhs, const ScalarVariable &gamma, VectorVariable &solution, const DimensionsHandlerVector<StrideFunc> &dim_hand)
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
    for (Dim comp = 0; comp < 3; ++comp)
    {
        for (Dim inedx_1 = 0; inedx_1 < N2; ++ ++inedx_1)
        {
            for (Dim inedx_2 = 0; inedx_2 < N3; ++inedx_2)
            {

                Dim stride = stride_func(inedx_1, inedx_2);

                for (Dim index_0 = 1; index_0 < N1 - 1; ++index_0)
                {
                    d[index_0] = rhs.get(comp, stride + index_0);

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
                    solution.set(comp, stride + index_0) = x[index_0];
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

                a[N1 - 1] = -gamma.get(stride + N1 - 1) / (dN1 * dN1);
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
}

template <typename StrideFunc>
void NavierStokesBrinkmann::block_solver(const ScalarVariable &rhs, ScalarVariable &solution, const DimensionsHandlerScalar<StrideFunc> &dim_hand)
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

void NavierStokesBrinkmann::solve()
{
    // Main solver routine

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


    for (Real t = 0.0f; t < T; t += dt)
    {
        pressure_predictor = pressure_solution + other_phi;
        gradient_pressure.set(0) = pressure_predictor.getGradient_x();
        gradient_pressure.set(1) = pressure_predictor.getGradient_y();
        gradient_pressure.set(2) = pressure_predictor.getGradient_z();
        compute_vector_g();
        compute_vector_xi();

        vector_rhs = xi - eta;
        impose_Dirichlet<decltype(stride_x)>(vector_rhs, x_vector_handler);
        block_solver<decltype(stride_x)>(vector_rhs, gamma_field, delta_eta, x_vector_handler);
        eta += delta_eta;


        vector_rhs = eta - zeta;
        impose_Dirichlet<decltype(stride_y)>(vector_rhs, y_vector_handler);
        block_solver<decltype(stride_y)>(eta - zeta, gamma_field, delta_zeta, y_vector_handler);
        zeta += delta_zeta;

        vector_rhs = zeta - velocity_solution;
        impose_Dirichlet<decltype(stride_z)>(vector_rhs, z_vector_handler);
        block_solver<decltype(stride_z)>(zeta - velocity_solution, gamma_field, delta_u, z_vector_handler);
        velocity_solution += delta_u;

        // Pressure solve
        compute_scalar_rhs_pressure();
        impose_Neumann<decltype(stride_x)>(rhs, x_scalar_handler);
        block_solver<decltype(stride_x)>(rhs, psi, x_scalar_handler);

        rhs = psi;
        impose_Neumann<decltype(stride_y)>(rhs, y_scalar_handler);
        block_solver<decltype(stride_y)>(rhs, phi, y_scalar_handler);

        rhs = phi;
        impose_Neumann<decltype(stride_z)>(rhs, z_scalar_handler);
        block_solver<decltype(stride_z)>(rhs, other_phi, z_scalar_handler);

        pressure_solution += other_phi;
    }
};