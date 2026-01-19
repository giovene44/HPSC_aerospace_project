#include "navier_stokes_brinkman.hpp"
#include <stdexcept>

#include <cmath> // for std::abs
#include <fstream>
#include <sstream>
#include <helper.hpp>
#include <filesystem>
#include <chrono>

Real NavierStokesBrinkmann::compute_beta(Dim i, Dim j, Dim k) const
{
    Real k_val = k_field.get(i, j, k);
    if (std::fabs(k_val) < 1e-12f)
        k_val = 1e-12f;

    return (dt * nu) / (Real(2.0) * k_val);
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
        for (Dim idy = 0; idy < Ny; ++idy)
        {
            for (Dim idz = 0; idz < Nz; ++idz)
            {
                Dim weight = 1;
                if (idx == 0 || idx == Nx - 1)
                    weight *= 0.5;
                if (idy == 0 || idy == Ny - 1)
                    weight *= 0.5;
                if (idz == 0 || idz == Nz - 1)
                    weight *= 0.5;
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
    if (std::fabs(k_val) < Real(1e-12))
        k_val = Real(1e-12);

    Real beta = compute_beta(i, j, k);
    return (dt * nu / Real(2.0)) / (Real(1.0) + beta);
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

    // Define a target index for debugging prints to avoid console flood
    /*
     constexpr Dim DEBUG_I = 1;
    constexpr Dim DEBUG_J = 1;
    constexpr Dim DEBUG_K = 1;
    constexpr Dim DEBUG_COMP = 0; // Check the x-component

    */

    g.set_all(Real(0.0));

    for (Dim comp = 0; comp < vector_rhs.size(); ++comp)
    {
        for (Dim idx = 0; idx < g.elements_per_component(); ++idx)
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
            Real dxx_eta = velocity_solution.second_derivative(comp, 0, idx);  // ∂²η/∂x²
            Real dyy_zeta = velocity_solution.second_derivative(comp, 1, idx); // ∂²ζ/∂y²
            Real dzz_u = velocity_solution.second_derivative(comp, 2, idx);    // ∂²u/∂z²
            Real laplacian = dxx_eta + dyy_zeta + dzz_u;

            // -----------------------------------------------------------------
            // Physical properties
            // -----------------------------------------------------------------
            Real k_val = std::max(k_field.get(idx), Real(1e-12)); // local permeability k (unused)

            // Evaluate forcing function
            std::vector<Real> forcing_current = {forcing_function.value<0>(x, y, z, t + dt * Real(0.5)),
                                                 forcing_function.value<1>(x, y, z, t + dt * Real(0.5)),
                                                 forcing_function.value<2>(x, y, z, t + dt * Real(0.5))};
            Real forcing = forcing_current[comp];

            // Real p_grad = gradient_pressure_predictor.value(comp, idx); // unused
            // Real velocity = velocity_solution.value(comp, idx); // unused

            // =================================================================

            // -----------------------------------------------------------------
            // Assemble RHS term
            // -----------------------------------------------------------------

            if (i == 0 || j == 0 || k == 0 || i == Nx - 1 || j == Ny - 1 || k == Nz - 1)
            {
                // Boundary points: set g to zero (or handle as needed)
                g.set(comp, idx) = forcing - nu / (Real(2.0) * k) * velocity_solution.value(comp, idx);
                continue;
            }

            Real g_val =
                forcing                                                           // f
                - gradient_pressure_predictor.value(comp, idx)                    // -∇p
                + Real(0.5) * nu * laplacian                                      // + (ν/2)(∇²η + ∇²ζ + ∇²u)
                - (nu / (Real(2.0) * k_val)) * velocity_solution.value(comp, idx) // - (ν/(2k))u₀
                ;

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
                (velocity_solution.value(comp, idx) + dt * g.value(comp, idx)) / compute_beta(idx);
        }
    }
}

void NavierStokesBrinkmann::compute_divergence_cell_center(const VectorVariable &u,
                                                           ScalarVariable &div)
{
    div.set_all(Real(0.0));

    for (Dim k = 1; k < Nz - 1; ++k)
        for (Dim j = 1; j < Ny - 1; ++j)
            for (Dim i = 1; i < Nx - 1; ++i)
            {
                const Real dudx = (u.value(0, i, j, k) - u.value(0, i - 1, j, k)) / dx;
                const Real dvdy = (u.value(1, i, j, k) - u.value(1, i, j - 1, k)) / dy;
                const Real dwdz = (u.value(2, i, j, k) - u.value(2, i, j, k - 1)) / dz;

                div.set(i, j, k) = dudx + dvdy + dwdz;
            }
}

void NavierStokesBrinkmann::compute_rhs_pressure(Real t)
{

    ScalarVariable div_u(Nx, Ny, Nz, dx, dy, dz);
    compute_divergence_cell_center(velocity_solution, div_u);
    rhs.set_all(Real(0.0));

    for (Dim k = 1; k < Nz - 1; ++k)
        for (Dim j = 1; j < Ny - 1; ++j)
            for (Dim i = 1; i < Nx - 1; ++i)
            {

                rhs.set(i, j, k) =
                    -div_u.get(i, j, k) / dt;
            }
}

static void compute_forcing_analytic(VectorVariable &f,
                                     Real dx, Real dy, Real dz,
                                     Dim Nx, Dim Ny, Dim Nz,
                                     Real t,
                                     Real nu,
                                     ScalarVariable k)
{
    f.set_all(Real(0.0));

    const Real A = std::sin(t);
    const Real Ap = std::cos(t);

    for (Dim kk = 1; kk < Nz - 1; ++kk)
        for (Dim jj = 1; jj < Ny - 1; ++jj)
            for (Dim ii = 1; ii < Nx - 1; ++ii)
            {
                // --------------------------------------------------
                // Component 0: u at (x+dx/2, y, z)
                // u = sin(t)*sin(x)*sin(y)*sin(z)
                // --------------------------------------------------
                {
                    const Real x = (Real(ii) + Real(0.5)) * dx;
                    const Real y = Real(jj) * dy;
                    const Real z = Real(kk) * dz;

                    const Real sx = std::sin(x), cx = std::cos(x);
                    const Real sy = std::sin(y), cy = std::cos(y);
                    const Real sz = std::sin(z), cz = std::cos(z);

                    const Real u = A * sx * sy * sz;
                    const Real ut = Ap * sx * sy * sz;

                    // Laplacian: d²u/dx² + d²u/dy² + d²u/dz²
                    const Real lap_u = -A * sx * sy * sz - A * sx * sy * sz - A * sx * sy * sz;

                    // Pressure gradient: dp/dx for p = sin(t)*cos(x)*cos(y)*cos(z)
                    const Real dp_dx = -std::sin(t) * std::sin(x) * std::cos(y) * std::cos(z);

                    f.set(0, ii, jj, kk) = ut - nu * lap_u + (nu / k.get(ii, jj, kk)) * u + dp_dx;
                }

                // --------------------------------------------------
                // Component 1: v at (x, y+dy/2, z)
                // v = sin(t)*cos(x)*cos(y)*cos(z)
                // --------------------------------------------------
                {
                    const Real x = Real(ii) * dx;
                    const Real y = (Real(jj) + Real(0.5)) * dy;
                    const Real z = Real(kk) * dz;

                    const Real sx = std::sin(x), cx = std::cos(x);
                    const Real sy = std::sin(y), cy = std::cos(y);
                    const Real sz = std::sin(z), cz = std::cos(z);

                    const Real v = A * cx * cy * cz;
                    const Real vt = Ap * cx * cy * cz;

                    // Laplacian
                    const Real lap_v = -A * cx * cy * cz - A * cx * cy * cz - A * cx * cy * cz;

                    // dp/dy = -sin(t)*cos(x)*sin(y)*cos(z)
                    const Real dp_dy = -std::sin(t) * std::cos(x) * std::sin(y) * std::cos(z);

                    f.set(1, ii, jj, kk) = vt - nu * lap_v + (nu / k.get(ii, jj, kk)) * v + dp_dy;
                }

                // --------------------------------------------------
                // Component 2: w at (x, y, z+dz/2)
                // w = sin(t)*cos(x)*sin(y)*(cos(z)+sin(z))
                // --------------------------------------------------
                {
                    const Real x = Real(ii) * dx;
                    const Real y = Real(jj) * dy;
                    const Real z = (Real(kk) + Real(0.5)) * dz;

                    const Real sx = std::sin(x), cx = std::cos(x);
                    const Real sy = std::sin(y), cy = std::cos(y);
                    const Real sz = std::sin(z), cz = std::cos(z);

                    const Real w = A * cx * sy * (cz + sz);
                    const Real wt = Ap * cx * sy * (cz + sz);

                    // Laplacian
                    const Real lap_w = -A * cx * sy * (cz + sz) - A * cx * sy * (cz + sz) - A * cx * sy * (cz + sz);

                    // dp/dz = -sin(t)*cos(x)*cos(y)*sin(z)
                    const Real dp_dz = -std::sin(t) * std::cos(x) * std::cos(y) * std::sin(z);

                    f.set(2, ii, jj, kk) = wt - nu * lap_w + (nu / k.get(ii, jj, kk)) * w + dp_dz;
                }
            }
}

static void build_rhs(VectorVariable &rhs,
                      const VectorVariable &velocity_solution,
                      const ScalarVariable &p_star, // predictor pressure at cell centers
                      const VectorVariable &f_half,
                      Real dx, Real dy, Real dz,
                      Dim Nx, Dim Ny, Dim Nz,
                      Real dt,
                      Real nu,
                      const ScalarVariable &k)
{
    for (int c = 0; c < 3; ++c)
        for (Dim kk = 0; kk < Nz; ++kk)
            for (Dim jj = 0; jj < Ny; ++jj)
                for (Dim ii = 0; ii < Nx; ++ii)
                {
                    const Real beta = (nu * dt) / (Real(2.0) * k.get(ii, jj, kk));
                    const Real scale = Real(1.0) / (Real(1.0) + beta);
                    const Real diff = (nu * dt) / Real(2.0);

                    Real val = velocity_solution.value(c, ii, jj, kk) + dt * f_half.value(c, ii, jj, kk) - beta * velocity_solution.value(c, ii, jj, kk);

                    const bool interior =
                        (ii > 0 && ii < Nx - 1 &&
                         jj > 0 && jj < Ny - 1 &&
                         kk > 0 && kk < Nz - 1);
                    if (interior)
                    {
                        // Explicit CN half diffusion: + (nu dt/2) Lap(u^n)
                        const Real u0 = velocity_solution.value(c, ii, jj, kk);

                        const Real uxx =
                            (velocity_solution.value(c, ii + 1, jj, kk) - Real(2.0) * u0 + velocity_solution.value(c, ii - 1, jj, kk)) / (dx * dx);
                        const Real uyy =
                            (velocity_solution.value(c, ii, jj + 1, kk) - Real(2.0) * u0 + velocity_solution.value(c, ii, jj - 1, kk)) / (dy * dy);
                        const Real uzz =
                            (velocity_solution.value(c, ii, jj, kk + 1) - Real(2.0) * u0 + velocity_solution.value(c, ii, jj, kk - 1)) / (dz * dz);

                        val += diff * (uxx + uyy + uzz);

                        // Predictor pressure gradient at staggered locations (MAC-consistent one-sided):
                        // u-face: dp/dx ≈ (p(i+1)-p(i))/dx
                        // v-face: dp/dy ≈ (p(j+1)-p(j))/dy
                        // w-face: dp/dz ≈ (p(k+1)-p(k))/dz
                        const Real dp_dx = (p_star.get(ii + 1, jj, kk) - p_star.get(ii, jj, kk)) / dx;
                        const Real dp_dy = (p_star.get(ii, jj + 1, kk) - p_star.get(ii, jj, kk)) / dy;
                        const Real dp_dz = (p_star.get(ii, jj, kk + 1) - p_star.get(ii, jj, kk)) / dz;

                        if (c == 0)
                            val -= dt * dp_dx;
                        else if (c == 1)
                            val -= dt * dp_dy;
                        else
                            val -= dt * dp_dz;
                    }

                    rhs.set(c, ii, jj, kk) = val * scale;
                }
}

#ifndef USE_MPI
Real NavierStokesBrinkmann::solve(const ManufacturedSolution &mms, bool openMP)
{
    Real t = Real(1.5);

    (void)mms; // Unused parameter
    DimensionsHandlerScalar x_scalar_handler(Nx, Ny, Nz, dx);
    DimensionsHandlerScalar y_scalar_handler(Ny, Nx, Nz, dy);
    DimensionsHandlerScalar z_scalar_handler(Nz, Nx, Ny, dz);

    DimensionsHandlerVector x_vector_handler(Nx, Ny, Nz, 0, 1, 2, dx);
    DimensionsHandlerVector y_vector_handler(Ny, Nx, Nz, 1, 0, 2, dy);
    DimensionsHandlerVector z_vector_handler(Nz, Nx, Ny, 2, 0, 1, dz);

    VectorVariable f_half(Nx, Ny, Nz, dx, dy, dz);

    // Initialize
    velocity_solution.set_all(u_boundary, t);
    pressure_solution.set_all(p_exact, t);

    // Init intermediate vars to avoid junk values
    xi = eta = zeta = velocity_solution;
    other_phi = phi = pressure_solution;
    // other_phi.set_all(Real(0.0));

    velocity_time_series.clear();
    pressure_time_series.clear();
    velocity_time_series.emplace_back(velocity_solution);
    pressure_time_series.emplace_back(pressure_solution);

    // write_pressure_vtk("./Output/pressure_N"+std::to_string(Nx) +"_step"+ std::to_string(0) + ".vtk");

    Dim nsteps = static_cast<Dim>(std::ceil(T / dt));
    // --- Time Stepping Loop ---

    // ScalarVariable corr_prev(Nx, Ny, Nz, dx, dy, dz);
    ScalarVariable div_u(Nx, Ny, Nz, dx, dy, dz);
    VectorVariable u_tmp(Nx, Ny, Nz, dx, dy, dz);
    VectorVariable u_np1(Nx, Ny, Nz, dx, dy, dz);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int n = 0; n < nsteps; ++n)
    {
        const Real t_np1 = t + dt;
        const Real t_half = t + dt / Real(2.0);

        pressure_predictor = pressure_solution + other_phi;

        // ---- Momentum step (ADI) with predictor pressure
        velocity_solver.set_t(t_np1);

        compute_forcing_analytic(f_half, dx, dy, dz, Nx, Ny, Nz, t_half, nu, k_field);

        // RHS uses (pressure_solution - ∇pressure_predictor) in an Auteri-consistent way
        build_rhs(xi, velocity_solution, pressure_predictor, f_half, dx, dy, dz, Nx, Ny, Nz, dt, nu, k_field);

        vector_rhs = xi - eta.A_operator(0, 0, gamma_field);
        velocity_solver.solve<0>(vector_rhs, eta, x_vector_handler, openMP);

        vector_rhs = eta - zeta.A_operator(1, 1, gamma_field);
        velocity_solver.solve<1>(vector_rhs, zeta, y_vector_handler, openMP);

        vector_rhs = zeta - velocity_solution.A_operator(2, 2, gamma_field);
        velocity_solver.solve<2>(vector_rhs, velocity_solution, z_vector_handler, openMP);

        // ---- Pressure correction (space-factored operator A)

        compute_rhs_pressure(t_np1);

        pressure_solver.set_t(t_np1);

        // (I - dxx) psi = rhs_p
        // (I - dyy) phi = psi
        // (I - dzz) corr_new = phi

        pressure_solver.solve_x(rhs, psi, openMP);
        pressure_solver.solve_y(psi, phi, openMP);
        pressure_solver.solve_z(phi, other_phi, openMP);

        // ---- Pressure update at half-step (Auteri):
        // p^{n+1/2} = p^{n-1/2} + ϕ^{n+1/2}

        pressure_solution += other_phi;

        t = t_np1;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    return elapsed.count();
}
#endif

void NavierStokesBrinkmann::write_velocity_vtk(const std::string &filename) const
{
    // Ensure the output directory exists
    std::filesystem::path filepath(filename);
    std::filesystem::create_directories(filepath.parent_path());

    std::ofstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file: " + filename);

    // VTK header
    file << "# vtk DataFile Version 3.0\n";
    file << "Velocity field\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_POINTS\n";
    file << "DIMENSIONS " << Nx << " " << Ny << " " << Nz << "\n";
    file << "SPACING " << dx << " " << dy << " " << dz << "\n";
    file << "ORIGIN 0 0 0\n";
    file << "POINT_DATA " << (Nx * Ny * Nz) << "\n";
    file << "VECTORS velocity float\n";

    // Write velocity data
    for (Dim idx = 0; idx < Nx * Ny * Nz; ++idx)
    {
        Real u = velocity_solution.value(0, idx);
        Real v = velocity_solution.value(1, idx);
        Real w = velocity_solution.value(2, idx);
        file << u << " " << v << " " << w << "\n";
    }

    file.close();
}

void NavierStokesBrinkmann::write_pressure_vtk(const std::string &filename) const
{
    // Ensure the output directory exists
    std::filesystem::path filepath(filename);
    std::filesystem::create_directories(filepath.parent_path());

    std::ofstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file: " + filename);

    // VTK header
    file << "# vtk DataFile Version 3.0\n";
    file << "Pressure field\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_POINTS\n";
    file << "DIMENSIONS " << Nx << " " << Ny << " " << Nz << "\n";
    file << "SPACING " << dx << " " << dy << " " << dz << "\n";
    file << "ORIGIN 0 0 0\n";
    file << "POINT_DATA " << (Nx * Ny * Nz) << "\n";
    file << "SCALARS pressure float 1\n";
    file << "LOOKUP_TABLE default\n";

    // Write pressure data
    for (Dim idx = 0; idx < Nx * Ny * Nz; ++idx)
    {
        Real p = pressure_solution.get(idx);
        file << p << "\n";
    }

    file.close();
}

#ifdef USE_MPI
// Helper function to synchronize scalar field across all MPI ranks
static void sync_scalar_field_full(ScalarVariable &field, Dim Nx, Dim Ny, Dim Nz,
                                   const MPITopology3D &topo)
{
    MPI_Comm comm = topo.cart_comm();
    int world_rank;
    MPI_Comm_rank(comm, &world_rank);

    const int total = Nx * Ny * Nz;

    std::vector<Real> local_data(total);
    for (Dim k = 0; k < Nz; ++k)
        for (Dim j = 0; j < Ny; ++j)
            for (Dim i = 0; i < Nx; ++i)
            {
                int idx = i + j * Nx + k * Nx * Ny;
                local_data[idx] = field.get(i, j, k);
            }

    auto block_range = [](Dim Nglobal, int p, int P) -> std::pair<Dim, Dim>
    {
        const Dim q = Nglobal / Dim(P);
        const Dim r = Nglobal % Dim(P);
        Dim begin, end;
        if (Dim(p) < r)
        {
            begin = Dim(p) * (q + 1);
            end = begin + (q + 1);
        }
        else
        {
            begin = r * (q + 1) + (Dim(p) - r) * q;
            end = begin + q;
        }
        return {begin, end};
    };

    if (world_rank == 0)
    {
        int world_size;
        MPI_Comm_size(comm, &world_size);

        for (int src = 1; src < world_size; ++src)
        {
            int src_coords[3];
            MPI_Cart_coords(comm, src, 3, src_coords);

            auto [i0, i1] = block_range(Nx, src_coords[2], topo.Px());
            auto [j0, j1] = block_range(Ny, src_coords[1], topo.Py());
            auto [k0, k1] = block_range(Nz, src_coords[0], topo.Pz());

            Dim ni = i1 - i0;
            Dim nj = j1 - j0;
            Dim nk = k1 - k0;
            int count = ni * nj * nk;

            std::vector<Real> recv_buf(count);
            MPI_Recv(recv_buf.data(), count, MPI_FLOAT, src, 0, comm, MPI_STATUS_IGNORE);

            int idx = 0;
            for (Dim kk = k0; kk < k1; ++kk)
                for (Dim jj = j0; jj < j1; ++jj)
                    for (Dim ii = i0; ii < i1; ++ii)
                    {
                        int gidx = ii + jj * Nx + kk * Nx * Ny;
                        local_data[gidx] = recv_buf[idx++];
                    }
        }
    }
    else
    {
        int my_coords[3];
        MPI_Cart_coords(comm, world_rank, 3, my_coords);

        auto [i0, i1] = block_range(Nx, my_coords[2], topo.Px());
        auto [j0, j1] = block_range(Ny, my_coords[1], topo.Py());
        auto [k0, k1] = block_range(Nz, my_coords[0], topo.Pz());

        Dim ni = i1 - i0;
        Dim nj = j1 - j0;
        Dim nk = k1 - k0;
        int count = ni * nj * nk;

        std::vector<Real> send_buf(count);
        int idx = 0;
        for (Dim kk = k0; kk < k1; ++kk)
            for (Dim jj = j0; jj < j1; ++jj)
                for (Dim ii = i0; ii < i1; ++ii)
                    send_buf[idx++] = field.get(ii, jj, kk);

        MPI_Send(send_buf.data(), count, MPI_FLOAT, 0, 0, comm);
    }

    MPI_Bcast(local_data.data(), total, MPI_FLOAT, 0, comm);

    for (Dim k = 0; k < Nz; ++k)
        for (Dim j = 0; j < Ny; ++j)
            for (Dim i = 0; i < Nx; ++i)
            {
                int idx = i + j * Nx + k * Nx * Ny;
                field.set(i, j, k) = local_data[idx];
            }
}

// Helper function to synchronize vector field across all MPI ranks
static void sync_vector_field_full(VectorVariable &field, Dim Nx, Dim Ny, Dim Nz,
                                   const MPITopology3D &topo)
{
    MPI_Comm comm = topo.cart_comm();
    int world_rank;
    MPI_Comm_rank(comm, &world_rank);

    const int total = Nx * Ny * Nz;

    auto block_range = [](Dim Nglobal, int p, int P) -> std::pair<Dim, Dim>
    {
        const Dim q = Nglobal / Dim(P);
        const Dim r = Nglobal % Dim(P);
        Dim begin, end;
        if (Dim(p) < r)
        {
            begin = Dim(p) * (q + 1);
            end = begin + (q + 1);
        }
        else
        {
            begin = r * (q + 1) + (Dim(p) - r) * q;
            end = begin + q;
        }
        return {begin, end};
    };

    for (int comp = 0; comp < 3; ++comp)
    {
        std::vector<Real> local_data(total);
        for (Dim k = 0; k < Nz; ++k)
            for (Dim j = 0; j < Ny; ++j)
                for (Dim i = 0; i < Nx; ++i)
                {
                    int idx = i + j * Nx + k * Nx * Ny;
                    local_data[idx] = field.value(comp, i, j, k);
                }

        if (world_rank == 0)
        {
            int world_size;
            MPI_Comm_size(comm, &world_size);

            for (int src = 1; src < world_size; ++src)
            {
                int src_coords[3];
                MPI_Cart_coords(comm, src, 3, src_coords);

                auto [i0, i1] = block_range(Nx, src_coords[2], topo.Px());
                auto [j0, j1] = block_range(Ny, src_coords[1], topo.Py());
                auto [k0, k1] = block_range(Nz, src_coords[0], topo.Pz());

                Dim ni = i1 - i0;
                Dim nj = j1 - j0;
                Dim nk = k1 - k0;
                int count = ni * nj * nk;

                std::vector<Real> recv_buf(count);
                MPI_Recv(recv_buf.data(), count, MPI_FLOAT, src, comp, comm, MPI_STATUS_IGNORE);

                int idx = 0;
                for (Dim kk = k0; kk < k1; ++kk)
                    for (Dim jj = j0; jj < j1; ++jj)
                        for (Dim ii = i0; ii < i1; ++ii)
                        {
                            int gidx = ii + jj * Nx + kk * Nx * Ny;
                            local_data[gidx] = recv_buf[idx++];
                        }
            }
        }
        else
        {
            int my_coords[3];
            MPI_Cart_coords(comm, world_rank, 3, my_coords);

            auto [i0, i1] = block_range(Nx, my_coords[2], topo.Px());
            auto [j0, j1] = block_range(Ny, my_coords[1], topo.Py());
            auto [k0, k1] = block_range(Nz, my_coords[0], topo.Pz());

            Dim ni = i1 - i0;
            Dim nj = j1 - j0;
            Dim nk = k1 - k0;
            int count = ni * nj * nk;

            std::vector<Real> send_buf(count);
            int idx = 0;
            for (Dim kk = k0; kk < k1; ++kk)
                for (Dim jj = j0; jj < j1; ++jj)
                    for (Dim ii = i0; ii < i1; ++ii)
                        send_buf[idx++] = field.value(comp, ii, jj, kk);

            MPI_Send(send_buf.data(), count, MPI_FLOAT, 0, comp, comm);
        }

        MPI_Bcast(local_data.data(), total, MPI_FLOAT, 0, comm);

        for (Dim k = 0; k < Nz; ++k)
            for (Dim j = 0; j < Ny; ++j)
                for (Dim i = 0; i < Nx; ++i)
                {
                    int idx = i + j * Nx + k * Nx * Ny;
                    field.set(comp, i, j, k) = local_data[idx];
                }
    }
}

Real NavierStokesBrinkmann::solve_mpi(const ManufacturedSolution &mms, const MPITopology3D &topo, bool use_omp)
{
    Real t = Real(1.5);

    (void)mms;

    VectorVariable f_half(Nx, Ny, Nz, dx, dy, dz);

    // Initialize
    velocity_solution.set_all(u_boundary, t);
    pressure_solution.set_all(p_exact, t);

    // Init intermediate vars
    xi = eta = zeta = velocity_solution;
    other_phi = phi = pressure_solution;

    velocity_time_series.clear();
    pressure_time_series.clear();
    velocity_time_series.emplace_back(velocity_solution);
    pressure_time_series.emplace_back(pressure_solution);

    Dim nsteps = static_cast<Dim>(std::ceil(T / dt));

    VectorVariable u_tmp(Nx, Ny, Nz, dx, dy, dz);
    VectorVariable u_np1(Nx, Ny, Nz, dx, dy, dz);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int n = 0; n < nsteps; ++n)
    {
        const Real t_np1 = t + dt;
        const Real t_half = t + dt / Real(2.0);

        pressure_predictor = pressure_solution + other_phi;

        velocity_solver.set_t(t_np1);

        compute_forcing_analytic(f_half, dx, dy, dz, Nx, Ny, Nz, t_half, nu, k_field);

        build_rhs(xi, velocity_solution, pressure_predictor, f_half, dx, dy, dz, Nx, Ny, Nz, dt, nu, k_field);

        // MPI ADI velocity solves with synchronization
        velocity_solver.solve_x_only(xi, u_tmp, topo, use_omp);
        sync_vector_field_full(u_tmp, Nx, Ny, Nz, topo);

        velocity_solver.solve_y_only(u_tmp, u_np1, topo, use_omp);
        sync_vector_field_full(u_np1, Nx, Ny, Nz, topo);

        velocity_solver.solve_z_only(u_np1, velocity_solution, topo, use_omp);
        sync_vector_field_full(velocity_solution, Nx, Ny, Nz, topo);

        // Pressure correction
        compute_rhs_pressure(t_np1);

        pressure_solver.set_t(t_np1);

        // MPI ADI pressure solves with synchronization
        pressure_solver.solve_x_mpi(rhs, psi, topo);
        sync_scalar_field_full(psi, Nx, Ny, Nz, topo);

        pressure_solver.solve_y_mpi(psi, phi, topo);
        sync_scalar_field_full(phi, Nx, Ny, Nz, topo);

        pressure_solver.solve_z_mpi(phi, other_phi, topo);
        sync_scalar_field_full(other_phi, Nx, Ny, Nz, topo);

        pressure_solution += other_phi;

        t = t_np1;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    return elapsed.count();
}
#endif