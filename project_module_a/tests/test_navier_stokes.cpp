#include <cassert>
#include <iostream>
#include "navier_stokes_brinkman.hpp"

namespace
{
    void dimension_test()
    {
        const Dim Nx = 4, Ny = 5, Nz = 6;
        const float dt = 0.01f;
        const Real dx = 0.1f, dy = 0.1f, dz = 0.1f;

        std::cout << "Running dimension test..." << std::endl;

        NavierStokesBrinkmann nsb(Nx, Ny, Nz, dt, dx, dy, dz);

        // -------------------------------------------------------------
        // Grid parameters
        // -------------------------------------------------------------
        assert(nsb.Nx == Nx);
        assert(nsb.Ny == Ny);
        assert(nsb.Nz == Nz);
        assert(std::abs(nsb.dx - dx) < 1e-12);
        assert(std::abs(nsb.dy - dy) < 1e-12);
        assert(std::abs(nsb.dz - dz) < 1e-12);

        std::cout << "✅ Grid dimensions stored correctly." << std::endl;

        // -------------------------------------------------------------
        // Vector fields (3 components)
        // -------------------------------------------------------------
        auto check_vector = [&](const VectorVariable &v, const std::string &name)
        {
            assert(v.size() == 3);
            assert(v.elements_per_component() == Nx * Ny * Nz);
            assert(v.get_Nx() == Nx);
            assert(v.get_Ny() == Ny);
            assert(v.get_Nz() == Nz);
            std::cout << "  • " << name << " OK" << std::endl;
        };

        check_vector(nsb.u_0, "u_0");
        check_vector(nsb.f, "f");

        check_vector(nsb.g, "g");
        check_vector(nsb.vector_rhs, "vector_rhs");
        check_vector(nsb.velocity_solution, "velocity_solution");

        std::cout << "✅ All vector fields have correct dimensions." << std::endl;

        // -------------------------------------------------------------
        // Scalar fields
        // -------------------------------------------------------------
        auto check_scalar = [&](const ScalarVariable &s, const std::string &name)
        {
            assert(s.size() == Nx * Ny * Nz);
            std::cout << "  • " << name << " OK" << std::endl;
        };

        check_scalar(nsb.p_0, "p_0");
        check_scalar(nsb.k_field, "k_field");
        check_scalar(nsb.gamma_field, "gamma_field");
        check_scalar(nsb.rhs, "rhs");
        check_scalar(nsb.psi, "psi");
        check_scalar(nsb.phi, "phi");
        check_scalar(nsb.other_phi, "other_phi");
        check_scalar(nsb.sol_linear_system, "sol_linear_system");
        check_scalar(nsb.a, "a");
        check_scalar(nsb.b, "b");
        check_scalar(nsb.c, "c");
        check_scalar(nsb.nu, "nu");
        check_scalar(nsb.pressure_solution, "pressure_solution");

        std::cout << "✅ All scalar fields have correct sizes." << std::endl;

        // -------------------------------------------------------------
        // Initial zero values for scalar fields
        // -------------------------------------------------------------
        for (Dim index = 0; index < nsb.rhs.size(); ++index)
        {
            assert(nsb.rhs.get(index) == 0.0f);
            assert(nsb.p_0.get(index) == 0.0f);
        }

        std::cout << "✅ Scalar fields initialized to zero." << std::endl;

        // -------------------------------------------------------------
        // No crash = all dimension checks passed
        // -------------------------------------------------------------
        std::cout << "🎯 Dimension test passed successfully!" << std::endl;
    }

    void compute_vector_g_test_v_p_costant()
    {
        const Dim Nx = 4, Ny = 4, Nz = 4;
        const float dt = 0.01f;
        const Real dx = 0.1f, dy = 0.1f, dz = 0.1f;

        std::cout << "Running g function test..." << std::endl;

        NavierStokesBrinkmann nsb(Nx, Ny, Nz, dt, dx, dy, dz);

        // -----------------------------------------------------------------
        // 1. Initialize fields
        // -----------------------------------------------------------------

        // External forcing term: constant field f = (1, 2, 3)
        for (int cmp = 0; cmp < 3; ++cmp)
        {
            for (Dim k = 0; k < Nz; ++k)
                for (Dim j = 0; j < Ny; ++j)
                    for (Dim i = 0; i < Nx; ++i)
                    {
                        nsb.f.set(cmp, i, j, k) = static_cast<Real>(cmp + 1);
                        nsb.u_0.set(cmp, i, j, k) = 2.0f;    // uniform velocity field
                        nsb.eta_0.set(cmp, i, j, k) = 2.0f;  // uniform velocity field
                        nsb.zeta_0.set(cmp, i, j, k) = 2.0f; // uniform velocity field
                    }
        }

        // Material properties: constant fields
        for (Dim index = 0; index < nsb.k_field.size(); ++index)
        {
            nsb.k_field.set(index) = 1.0f; // permeability k = 1
            nsb.nu.set(index) = 1.0f;      // viscosity ν = 1
        }

        // -----------------------------------------------------------------
        // 2. Compute g field
        // -----------------------------------------------------------------
        nsb.compute_vector_g();

        // -----------------------------------------------------------------
        // 3. Check results
        // Expected: g = (0, 1, 2)
        // because g = f - (ν / (2k)) * u_0 = (1,2,3) - (1/2)*2 = (0,1,2)
        // -----------------------------------------------------------------
        for (int cmp = 0; cmp < 3; ++cmp)
        {
            Real expected_g = static_cast<Real>(cmp);
            for (Dim k = 1; k < Nz - 1; ++k)
                for (Dim j = 1; j < Ny - 1; ++j)
                    for (Dim i = 1; i < Nx - 1; ++i)
                    {
                        Real computed_g = nsb.g.value(cmp, i, j, k);
                        assert(std::abs(computed_g - expected_g) < 1e-6f);
                    }
        }

        std::cout << "✅ g function test passed successfully!" << std::endl;
    }

    void compute_vector_g_test_zero_forcing()
    {
        const Dim Nx = 4, Ny = 4, Nz = 4;
        const float dt = 0.01f;
        const Real dx = 0.1f, dy = 0.1f, dz = 0.1f;

        std::cout << "Running g test: zero forcing..." << std::endl;

        NavierStokesBrinkmann nsb(Nx, Ny, Nz, dt, dx, dy, dz);

        // f = 0
        for (int cmp = 0; cmp < 3; ++cmp)
            for (Dim k = 0; k < Nz; ++k)
                for (Dim j = 0; j < Ny; ++j)
                    for (Dim i = 0; i < Nx; ++i)
                        nsb.f.set(cmp, i, j, k) = 0.0f;

        // uniform velocity
        for (int cmp = 0; cmp < 3; ++cmp)
            for (Dim k = 0; k < Nz; ++k)
                for (Dim j = 0; j < Ny; ++j)
                    for (Dim i = 0; i < Nx; ++i)
                        nsb.u_0.set(cmp, i, j, k) = 2.0f;

        // material properties
        for (Dim idx = 0; idx < nsb.k_field.size(); ++idx)
        {
            nsb.k_field.set(idx) = 2.0f; // k = 2
            nsb.nu.set(idx) = 1.0f;      // ν = 1
        }

        nsb.compute_vector_g();

        // Expected: g = - (ν / (2k)) u₀ = - (1 / (2×2)) × 2 = -0.5
        for (int cmp = 0; cmp < 3; ++cmp)
        {
            Real expected_g = -0.5f * 1.0f; // same for all comps
            for (Dim k = 1; k < Nz - 1; ++k)
                for (Dim j = 1; j < Ny - 1; ++j)
                    for (Dim i = 1; i < Nx - 1; ++i)
                        assert(std::abs(nsb.g.value(cmp, i, j, k) - expected_g) < 1e-6f);
        }

        std::cout << "✅ g zero-forcing test passed." << std::endl;
    }

    void compute_vector_g_test_variable_force()
    {
        const Dim Nx = 4, Ny = 4, Nz = 4;
        const float dt = 0.01f;
        const Real dx = 0.1f, dy = 0.1f, dz = 0.1f;

        std::cout << "Running g test: variable forcing..." << std::endl;

        NavierStokesBrinkmann nsb(Nx, Ny, Nz, dt, dx, dy, dz);

        // f_x = i + j + k, f_y = 2*(i+j+k), f_z = 3*(i+j+k)
        for (Dim k = 0; k < Nz; ++k)
            for (Dim j = 0; j < Ny; ++j)
                for (Dim i = 0; i < Nx; ++i)
                {
                    Real val = static_cast<Real>(i + j + k);
                    nsb.f.set(0, i, j, k) = val;
                    nsb.f.set(1, i, j, k) = 2.0f * val;
                    nsb.f.set(2, i, j, k) = 3.0f * val;
                    nsb.u_0.set(0, i, j, k) = 2.0f;
                    nsb.u_0.set(1, i, j, k) = 2.0f;
                    nsb.u_0.set(2, i, j, k) = 2.0f;
                }

        for (Dim idx = 0; idx < nsb.k_field.size(); ++idx)
        {
            nsb.k_field.set(idx) = 1.0f;
            nsb.nu.set(idx) = 1.0f;
        }

        nsb.compute_vector_g();

        // Expected: g = f - (ν/(2k))*u₀ = f - 1 = (f_x-1, f_y-1, f_z-1)
        for (Dim k = 1; k < Nz - 1; ++k)
            for (Dim j = 1; j < Ny - 1; ++j)
                for (Dim i = 1; i < Nx - 1; ++i)
                {
                    Real val = static_cast<Real>(i + j + k);
                    assert(std::abs(nsb.g.value(0, i, j, k) - (val - 1.0f)) < 1e-6f);
                    assert(std::abs(nsb.g.value(1, i, j, k) - (2.0f * val - 1.0f)) < 1e-6f);
                    assert(std::abs(nsb.g.value(2, i, j, k) - (3.0f * val - 1.0f)) < 1e-6f);
                }

        std::cout << "✅ g variable-forcing test passed." << std::endl;
    }

    void compute_vector_g_test_diffusion_term()
    {
        const Dim Nx = 4, Ny = 4, Nz = 4;
        const float dt = 0.01f;
        const Real dx = 0.1f, dy = 0.1f, dz = 0.1f;

        std::cout << "Running g test: diffusion term..." << std::endl;

        NavierStokesBrinkmann nsb(Nx, Ny, Nz, dt, dx, dy, dz);

        // f = 0
        for (int cmp = 0; cmp < 3; ++cmp)
            for (Dim k = 0; k < Nz; ++k)
                for (Dim j = 0; j < Ny; ++j)
                    for (Dim i = 0; i < Nx; ++i)
                        nsb.f.set(cmp, i, j, k) = 0.0f;

        // velocity: u_x = x^2, others = 0
        for (Dim k = 0; k < Nz; ++k)
            for (Dim j = 0; j < Ny; ++j)
                for (Dim i = 0; i < Nx; ++i)
                {
                    Real x = i * dx;
                    nsb.u_0.set(0, i, j, k) = x * x;
                    nsb.u_0.set(1, i, j, k) = 0.0f;
                    nsb.u_0.set(2, i, j, k) = 0.0f;

                    nsb.eta_0.set(0, i, j, k) = x * x;
                    nsb.eta_0.set(1, i, j, k) = 0.0f;
                    nsb.eta_0.set(2, i, j, k) = 0.0f;

                    nsb.zeta_0.set(0, i, j, k) = x * x;
                    nsb.zeta_0.set(1, i, j, k) = 0.0f;
                    nsb.zeta_0.set(2, i, j, k) = 0.0f;
                }

        for (Dim idx = 0; idx < nsb.k_field.size(); ++idx)
        {
            nsb.k_field.set(idx) = 1.0f;
            nsb.nu.set(idx) = 1.0f;
        }

        nsb.compute_vector_g();

        // Expected: g_x = -(ν/(2k))*u + (ν/2)*∇²u = -0.5*u + 1.0
        for (Dim k = 1; k < Nz - 1; ++k)
            for (Dim j = 1; j < Ny - 1; ++j)
                for (Dim i = 1; i < Nx - 1; ++i)
                {
                    Real x = i * dx;
                    Real expected_gx = -0.5f * (x * x) + 1.0f;
                    assert(std::abs(nsb.g.value(0, i, j, k) - expected_gx) < 1e-3f);
                }

        std::cout << "✅ g diffusion-term test passed." << std::endl;
    }

    void compute_vector_xi_test_constant()
    {
        const Dim Nx = 4, Ny = 4, Nz = 4;
        const float dt = 0.01f;
        const Real dx = 0.1f, dy = 0.1f, dz = 0.1f;

        std::cout << "Running ξ test: constant fields..." << std::endl;

        NavierStokesBrinkmann nsb(Nx, Ny, Nz, dt, dx, dy, dz);

        // u₀ = 2, g = (1, 2, 3)
        for (int cmp = 0; cmp < 3; ++cmp)
            for (Dim k = 0; k < Nz; ++k)
                for (Dim j = 0; j < Ny; ++j)
                    for (Dim i = 0; i < Nx; ++i)
                    {
                        nsb.u_0.set(cmp, i, j, k) = 2.0f;
                        nsb.g.set(cmp, i, j, k) = static_cast<Real>(cmp + 1);
                    }

        // Material properties
        for (Dim idx = 0; idx < nsb.k_field.size(); ++idx)
        {
            nsb.k_field.set(idx) = 1.0f;
            nsb.nu.set(idx) = 1.0f;
        }

        // Compute ξ
        nsb.compute_vector_xi();

        // Expected: β = 1 + (dt*ν)/(2*k) = 1.005
        const Real beta = 1.0f + (dt * 1.0f) / (2.0f * 1.0f);

        // Expected ξ = u₀ + (Δt/β)*g
        for (int cmp = 0; cmp < 3; ++cmp)
        {
            Real expected_val = 2.0f + (dt / beta) * (cmp + 1);
            for (Dim k = 1; k < Nz - 1; ++k)
                for (Dim j = 1; j < Ny - 1; ++j)
                    for (Dim i = 1; i < Nx - 1; ++i)
                    {
                        Real val = nsb.xi.value(cmp, i, j, k);
                        assert(std::abs(val - expected_val) < 1e-6f);
                    }
        }

        std::cout << "✅ ξ constant-field test passed." << std::endl;
    }

    void compute_vector_xi_test_variable_g()
    {
        const Dim Nx = 4, Ny = 4, Nz = 4;
        const float dt = 0.01f;
        const Real dx = 0.1f, dy = 0.1f, dz = 0.1f;

        std::cout << "Running ξ test: variable g..." << std::endl;

        NavierStokesBrinkmann nsb(Nx, Ny, Nz, dt, dx, dy, dz);

        // u₀ = 2 constant, g = i + j + k + cmp
        for (Dim k = 0; k < Nz; ++k)
            for (Dim j = 0; j < Ny; ++j)
                for (Dim i = 0; i < Nx; ++i)
                {
                    Real base = static_cast<Real>(i + j + k);
                    for (int cmp = 0; cmp < 3; ++cmp)
                    {
                        nsb.u_0.set(cmp, i, j, k) = 2.0f;
                        nsb.g.set(cmp, i, j, k) = base + cmp;
                    }
                }

        // constant material properties
        for (Dim idx = 0; idx < nsb.k_field.size(); ++idx)
        {
            nsb.k_field.set(idx) = 1.0f;
            nsb.nu.set(idx) = 1.0f;
        }

        nsb.compute_vector_xi();

        const Real beta = 1.0f + (dt * 1.0f) / (2.0f * 1.0f); // 1.005

        // ξ = 2 + (dt/β)*(g)
        for (Dim k = 1; k < Nz - 1; ++k)
            for (Dim j = 1; j < Ny - 1; ++j)
                for (Dim i = 1; i < Nx - 1; ++i)
                    for (int cmp = 0; cmp < 3; ++cmp)
                    {
                        Real base = static_cast<Real>(i + j + k);
                        Real expected = 2.0f + (dt / beta) * (base + cmp);
                        Real actual = nsb.xi.value(cmp, i, j, k);
                        assert(std::abs(actual - expected) < 1e-6f);
                    }

        std::cout << "✅ ξ variable-g test passed." << std::endl;
    }

    void compute_vector_xi_test_variable_k()
    {
        const Dim Nx = 4, Ny = 4, Nz = 4;
        const float dt = 0.01f;
        const Real dx = 0.1f, dy = 0.1f, dz = 0.1f;

        std::cout << "Running ξ test: variable k..." << std::endl;

        NavierStokesBrinkmann nsb(Nx, Ny, Nz, dt, dx, dy, dz);

        // u₀ = 2, g = 1 for all components
        for (int cmp = 0; cmp < 3; ++cmp)
            for (Dim k = 0; k < Nz; ++k)
                for (Dim j = 0; j < Ny; ++j)
                    for (Dim i = 0; i < Nx; ++i)
                    {
                        nsb.u_0.set(cmp, i, j, k) = 2.0f;
                        nsb.g.set(cmp, i, j, k) = 1.0f;
                    }

        // variable k
        for (Dim k = 0; k < Nz; ++k)
            for (Dim j = 0; j < Ny; ++j)
                for (Dim i = 0; i < Nx; ++i)
                {
                    Real kval = 1.0f + 0.5f * (i + j + k);
                    nsb.k_field.set(i + j * Nx + k * Nx * Ny) = kval;
                    nsb.nu.set(i + j * Nx + k * Nx * Ny) = 1.0f;
                }

        nsb.compute_vector_xi();

        // ξ = u₀ + (Δt/β)*g with β = 1 + (Δt·ν)/(2·k)
        for (Dim k = 1; k < Nz - 1; ++k)
            for (Dim j = 1; j < Ny - 1; ++j)
                for (Dim i = 1; i < Nx - 1; ++i)
                {
                    Real kval = 1.0f + 0.5f * (i + j + k);
                    Real beta = 1.0f + (dt * 1.0f) / (2.0f * kval);
                    Real expected = 2.0f + (dt / beta) * 1.0f;
                    for (int cmp = 0; cmp < 3; ++cmp)
                        assert(std::abs(nsb.xi.value(cmp, i, j, k) - expected) < 1e-6f);
                }

        std::cout << "✅ ξ variable-k test passed." << std::endl;
    }

    void compute_vector_xi_test_zero_g()
    {
        const Dim Nx = 4, Ny = 4, Nz = 4;
        const float dt = 0.01f;
        const Real dx = 0.1f, dy = 0.1f, dz = 0.1f;

        std::cout << "Running ξ test: zero g..." << std::endl;

        NavierStokesBrinkmann nsb(Nx, Ny, Nz, dt, dx, dy, dz);

        // u₀ = 2, g = 0
        for (int cmp = 0; cmp < 3; ++cmp)
            for (Dim k = 0; k < Nz; ++k)
                for (Dim j = 0; j < Ny; ++j)
                    for (Dim i = 0; i < Nx; ++i)
                    {
                        nsb.u_0.set(cmp, i, j, k) = 2.0f;
                        nsb.g.set(cmp, i, j, k) = 0.0f;
                    }

        // material props
        for (Dim idx = 0; idx < nsb.k_field.size(); ++idx)
        {
            nsb.k_field.set(idx) = 1.0f;
            nsb.nu.set(idx) = 1.0f;
        }

        nsb.compute_vector_xi();

        // Expected: ξ = u₀ = 2.0
        for (int cmp = 0; cmp < 3; ++cmp)
            for (Dim k = 1; k < Nz - 1; ++k)
                for (Dim j = 1; j < Ny - 1; ++j)
                    for (Dim i = 1; i < Nx - 1; ++i)
                    {
                        assert(std::abs(nsb.xi.value(cmp, i, j, k) - 2.0f) < 1e-6f);
                    }

        std::cout << "✅ ξ zero-g test passed." << std::endl;
    }

}

int main()
{
    try
    {
        dimension_test();
        compute_vector_g_test_v_p_costant();
        compute_vector_g_test_zero_forcing();
        compute_vector_g_test_variable_force();
        compute_vector_g_test_diffusion_term();
        compute_vector_xi_test_constant();
        compute_vector_xi_test_variable_g();
        compute_vector_xi_test_variable_k();
        compute_vector_xi_test_zero_g();
    }
    catch (const std::exception &e)
    {
        std::cerr << "❌ Exception: " << e.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << "❌ Unknown error occurred." << std::endl;
        return 1;
    }

    return 0;
}
