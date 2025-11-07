#include "../Solver.hpp"

class W_solver : public Solver
{
protected:
    void apply_bc_x_dir(ScalarVariable &rhs) override
    { // THIS IS NOT CORRECT we miss 4 cases
        // w at i-planes (ghost on i = Nx-1)
        for (Dim j = 0; j < Ny; ++j)
        {
            for (Dim k = 0; k < Nz; ++k)
            {
                rhs.set(0, j, k) = BC_w(0.0f, y_at(j), z_at(k));

                const float c = -static_cast<float>(gamma_field.get(Nx - 1, j, k));
                rhs.set(Nx - 1, j, k) = rhs.get(Nx - 1, j, k) - 2.0f * c * BC_w(x_at(Nx - 1), y_at(j), z_at(k));
            }
        }
    }

    void apply_bc_y_dir(ScalarVariable &rhs) override
    {

        // w at j-planes (ghost on j = Ny-1)
        for (Dim i = 0; i < Nx; ++i)
        {
            for (Dim k = 0; k < Nz; ++k)
            {
                rhs.set(i, 0, k) = BC_w(x_at(i), 0.0f, z_at(k));

                const float c = -static_cast<float>(gamma_field.get(i, Ny - 1, k));
                rhs.set(i, Ny - 1, k) = rhs.get(i, Ny - 1, k) - 2.0f * c * BC_w(x_at(i), y_at(Ny - 1), z_at(k));
            }
        }
    }

    void apply_bc_z_dir(ScalarVariable &rhs) override
    {
        // w at k = 0 : divergence-based
        for (Dim i = 0; i < Nx; ++i)
        {
            for (Dim j = 0; j < Ny; ++j)
            {
                const float du_dx = BC_u(x_half(i), y_at(j), 0.0f) - BC_u(x_at(i) - 0.5f * dx, y_at(j), 0.0f);
                const float dv_dy = BC_v(x_at(i), y_half(j), 0.0f) - BC_v(x_at(i), y_at(j) - 0.5f * dy, 0.0f);

                rhs.set(i, j, 0) = BC_w(x_at(i), y_at(j), 0.0f) + 0.5f * dz * (-du_dx / dx - dv_dy / dy);
                rhs.set(i, j, Nz - 1) = BC_w(x_at(i), y_at(j), z_at(Nz - 1));
            }
        }
    }

public:
    W_solver(Dim Nx_, Dim Ny_, Dim Nz_, Real dx_, Real dy_, Real dz_, ScalarVariable gam)
        : Solver(Nx_, Ny_, Nz_, dx_, dy_, dz_, gam)
    {
    }
    void solve_x_dir(ScalarVariable &rhs, ScalarVariable &solution) override
    {
        apply_bc_x_dir(rhs);
        block_solver_mom(rhs, solution, dim_hand_x);
    };
    void solve_y_dir(ScalarVariable &rhs, ScalarVariable &solution) override
    {
        apply_bc_y_dir(rhs);
        block_solver_mom(rhs, solution, dim_hand_y);
    };
    void solve_z_dir(ScalarVariable &rhs, ScalarVariable &solution) override
    {
        apply_bc_z_dir(rhs);
        block_solver_mom(rhs, solution, dim_hand_z);
    };
};