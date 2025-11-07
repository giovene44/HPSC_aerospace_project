#include "../Solver.hpp"

class U_solver : public Solver
{
protected:
    void apply_bc_x_dir(ScalarVariable &rhs) override
    { // THIS IS NOT CORRECT we miss 4 cases
        for (Dim j = 0; j < Ny; ++j)
        {
            for (Dim k = 0; k < Nz; ++k)
            {
                const float dv_dy = BC_v(0.0f, y_half(j), z_at(k)) - BC_v(0.0f, y_at(j) - 0.5f * dy, z_at(k));
                const float dw_dz = BC_w(0.0f, y_at(j), z_half(k)) - BC_w(0.0f, y_at(j), z_at(k) - 0.5f * dz);

                rhs.set(0, j, k) = BC_u(0.0f, y_at(j), z_at(k)) + 0.5f * dx * (-dv_dy / dy - dw_dz / dz);
                rhs.set(Nx - 1, j, k) = BC_u(x_at(Nx - 1), y_at(j), z_at(k));
            }
        }
    }

    void apply_bc_y_dir(ScalarVariable &rhs) override
    {
        // u at j-planes (ghost on j = Ny-1)
        for (Dim i = 0; i < Nx; ++i)
        {
            for (Dim k = 0; k < Nz; ++k)
            {
                rhs.set(i, 0, k) = BC_u(x_at(i), 0.0f, z_at(k));
                const float c = -static_cast<float>(gamma_field.get(i, Ny - 1, k));
                rhs.set(i, Ny - 1, k) = rhs.get(i, Ny - 1, k) - 2.0f * c * BC_u(x_at(i), y_at(Ny - 1), z_at(k));
            }
        }
    }

    void apply_bc_z_dir(ScalarVariable &rhs) override
    {
        // u at k-planes (ghost on k = Nz-1)
        for (Dim i = 0; i < Nx; ++i)
        {
            for (Dim j = 0; j < Ny; ++j)
            {
                rhs.set(i, j, 0) = BC_u(x_at(i), y_at(j), 0.0f);

                const float c = -static_cast<float>(gamma_field.get(i, j, Nz - 1));
                rhs.set(i, j, Nz - 1) = rhs.get(i, j, Nz - 1) - 2.0f * c * BC_u(x_at(i), y_at(j), z_at(Nz - 1));
            }
        }
    }

public:
    U_solver(Dim Nx_, Dim Ny_, Dim Nz_, Real dx_, Real dy_, Real dz_, ScalarVariable gam)
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