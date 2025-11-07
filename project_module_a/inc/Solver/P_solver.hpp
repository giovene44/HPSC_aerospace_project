#include "../Solver.hpp"

class P_solver : public Solver
{
protected:
    void apply_bc_x_dir(ScalarVariable &rhs) override
    {

        int i_0, i_n_1, j, k;
        float c, value_0, value_1;

        i_0 = 0;
        value_0 = 2.0f * dx * neumann_boundary_condition;

        i_n_1 = Nx - 1;
        c = (1.0f + 1.0f / dx * dx);
        value_1 = -dx * c * neumann_boundary_condition;

        for (j = 0; j < Ny; j++)
        {
            for (k = 0; k < Nz; k++)
            {
                rhs.set(i_0, j, k) = rhs.get(i_0, j, k) + value_0;
                rhs.set(i_n_1, j, k) = rhs.get(i_n_1, j, k) + value_1;
            }
        }
    }

    void apply_bc_y_dir(ScalarVariable &rhs) override
    {
        int j_0, j_n_1, i, k;
        float c, value_0, value_1;

        j_0 = 0;
        value_0 = 2.0f * dy * neumann_boundary_condition;

        j_n_1 = Ny - 1;
        c = (1.0f + 1.0f / (dy * dy));
        value_1 = -dy * c * neumann_boundary_condition;

        for (i = 0; i < Nx; i++)
        {
            for (k = 0; k < Nz; k++)
            {
                rhs.set(i, j_0, k) = rhs.get(i, j_0, k) + value_0;
                rhs.set(i, j_n_1, k) = rhs.get(i, j_n_1, k) + value_1;
            }
        }
    }

    void apply_bc_z_dir(ScalarVariable &rhs) override
    {
        int k_0, k_n_1, i, j;
        float c, value_0, value_1;

        k_0 = 0;
        value_0 = 2.0f * dz * neumann_boundary_condition;

        k_n_1 = Nz - 1;
        c = (1.0f + 1.0f / (dz * dz));
        value_1 = -dz * c * neumann_boundary_condition;

        for (i = 0; i < Nx; i++)
        {
            for (j = 0; j < Ny; j++)
            {
                rhs.set(i, j, k_0) = rhs.get(i, j, k_0) + value_0;
                rhs.set(i, j, k_n_1) = rhs.get(i, j, k_n_1) + value_1;
            }
        }
    }

public:
    P_solver(Dim Nx_, Dim Ny_, Dim Nz_, Real dx_, Real dy_, Real dz_, ScalarVariable gam)
        : Solver(Nx_, Ny_, Nz_, dx_, dy_, dz_, gam)
    {
    }
    void solve_x_dir(ScalarVariable &rhs, ScalarVariable &solution) override
    {
        apply_bc_x_dir(rhs);
        block_solver_press(rhs, solution, dim_hand_x);
    };
    void solve_y_dir(ScalarVariable &rhs, ScalarVariable &solution) override
    {
        apply_bc_y_dir(rhs);
        block_solver_press(rhs, solution, dim_hand_y);
    };
    void solve_z_dir(ScalarVariable &rhs, ScalarVariable &solution) override
    {
        apply_bc_z_dir(rhs);
        block_solver_press(rhs, solution, dim_hand_z);
    };

    float neumann_boundary_condition = 0.0f;
};