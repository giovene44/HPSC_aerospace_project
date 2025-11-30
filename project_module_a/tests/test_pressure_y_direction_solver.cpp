#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>
#include "navier_stokes_brinkman.hpp"

// ===============================================================
// 1. Setup Grid Parameters
// ===============================================================
struct Grid
{
    Dim Nx, Ny, Nz;
    Real dx, dy, dz;
    Real dt;
};

Grid setup_grid(Real dim_x, Real dim_y, Real dim_z, Dim Nx, Dim Ny, Dim Nz, Real dt)
{
    Grid g;
    g.Nx = Nx;
    g.Ny = Ny;
    g.Nz = Nz;
    g.dx = dim_x / (Nx - 0.5);
    g.dy = dim_y / (Ny - 0.5);
    g.dz = dim_z / (Nz - 0.5);
    g.dt = dt;
    return g;
}

// ===============================================================
// 3. Initialize Fields (interior + RHS)
// ===============================================================
void initialize_fields(const Grid &g,
                       ScalarVariable &scalar,
                       ScalarVariable &rhs,
                       BoundaryFunctions &p_boundary,
                       Real t)
{

    // Initialize interior
    for (int k = 0; k < g.Nz; ++k)
        for (int j = 0; j < g.Ny; ++j)
            for (int i = 0; i < g.Nx; ++i)
            {
                Real x, y, z;
                x = i * g.dx;
                y = j * g.dy;
                z = k * g.dz;

                scalar.set(i, j, k) = sin(y) * sin(t);
            }

    // Build RHS = (I - Dyy) * scalar
    // For Neumann BCs, RHS is computed everywhere; boundary gradients handled in solver
    for (Dim k = 0; k < g.Nz; ++k)
        for (Dim j = 0; j < g.Ny; ++j)
            for (Dim i = 0; i < g.Nx; ++i)
            {
                Real val = scalar.get(i, j, k);
                Real sd = scalar.second_derivative(1, i, j, k);
                Real rhs_val = val - sd;
                rhs.set(i, j, k) = rhs_val;
            }
}

// ===============================================================
// 4. Setup Solver & Strides
// ===============================================================
PressureSolver setup_solver(const Grid &g, BoundaryFunctions &p_boundary, Real t)
{
    PressureSolver solver(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz, t, p_boundary);
    solver.p_boundary = p_boundary;
    return solver;
}

auto define_stride_y(const Grid &g)
{
    return [=](Dim i, Dim k)
    { return i + k * g.Nx * g.Ny; };
}

// ===============================================================
// 6. Solve and check solution
// ===============================================================
bool solve_and_check(PressureSolver &solver, ScalarVariable &rhs,
                     ScalarVariable &scalar, const Grid &g,
                     DimensionsHandlerVector<decltype(define_stride_y(g))> &y_handler)
{

    ScalarVariable computed_sol(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    computed_sol.set_all(0.0);

    solver.solve_pressure<decltype(define_stride_y(g)), 1>(rhs, computed_sol, y_handler);

    Real tolerance = 1e-3;
    bool res = true;

    for (Dim j = 0; j < g.Ny; ++j)
        for (Dim k = 0; k < g.Nz; ++k)
            for (Dim i = 0; i < g.Nx; ++i)
                if (std::abs(scalar.get(i, j, k) - computed_sol.get(i, j, k)) > tolerance)
                {
                    printf("Mismatch (i,j,k)=(%d,%d,%d): expected %f, got %f\n",
                           i, j, k, scalar.get(i, j, k), computed_sol.get(i, j, k));
                    res = false;
                }

    return res;
}

// ===============================================================
// 7. Main function
// ===============================================================
int main()
{
    constexpr Real two_pi = 2.0 * M_PI;
    Grid g = setup_grid(two_pi, two_pi, two_pi, 10, 10, 10, 0.002f);

    printf("Grid setup: Nx=%d, Ny=%d, Nz=%d, dx=%f, dy=%f, dz=%f, dt=%f\n",
           g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz, g.dt);

    ScalarVariable scalar(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    ScalarVariable rhs(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);

    BoundaryFunctions p_boundary;
    // Neumann BC: ∂p/∂y = cos(y)*sin(t) at boundaries
    std::vector<std::string> neumann_bc = {
        "0",
        "cos(y)*sin(t)",
        "0"};

    p_boundary.set_string_expression(neumann_bc);

    initialize_fields(g, scalar, rhs, p_boundary, g.dt);
    PressureSolver solver = setup_solver(g, p_boundary, g.dt);

    auto stride_y = define_stride_y(g);
    DimensionsHandlerVector<decltype(stride_y)> y_handler(g.Nx, g.Ny, g.Nz, 1, 0, 2, g.dy, stride_y);

    bool success = solve_and_check(solver, rhs, scalar, g, y_handler);

    if (success)
        std::cout << "[PASS] Solver matrix and inversion are consistent.\n";
    else
        std::cout << "[FAIL] Solver test failed.\n";

    return success ? 0 : -1;
}
