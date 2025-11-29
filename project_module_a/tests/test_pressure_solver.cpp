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
                       ScalarVariable &vector,
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

                    vector.set(i, j, k) = -1.0 * sin(t) * 2 * 1e5 * cos(x) * sin(y) * (sin(z) - cos(z));
                }

    // Apply known faces
    // apply_known_faces(vector, u_boundary, g, gamma_field, 0.0);

    // Build RHS = (I - Dxx) * vector
        for (Dim k = 0; k < g.Nz; ++k)
            for (Dim j = 0; j < g.Ny; ++j)
                for (Dim i = 0; i < g.Nx; ++i)
                {
                    Real val = vector.get(i, j, k);
                    Real sd = vector.second_derivative(0, i, j, k);
                    Real rhs_val = val - sd;

                    // Apply boundary BCs on RHS

                    if (i == 0 || i == g.Nx - 1 )
                    {
                        Real x_coord, y_coord, z_coord;
                            x_coord = i * g.dx;
                            y_coord = j * g.dy;
                            z_coord = k * g.dz;

                            rhs_val = p_boundary.value<>(x_coord, y_coord, z_coord, t);
                            vector.set(i, j, k) = rhs_val;
                    }
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

auto define_stride_x(const Grid &g)
{
    return [=](Dim j, Dim k)
    { return j * g.Nx + k * g.Nx * g.Ny; };
}

// ===============================================================
// 5. Diagnostics: Check matrix operator & residual
// ===============================================================
void check_matrix_operator(PressureSolver &solver, const ScalarVariable &vector,
                           ScalarVariable &rhs, const Grid &g,
                           DimensionsHandlerVector<decltype(define_stride_x(g))> &x_handler)
{
    ScalarVariable Ax(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    solver.apply_matrix_operator<0, decltype(define_stride_x(g))>(vector, Ax, rhs, x_handler);
    std::cout << "Matrix operator diagnostic on boundaries and interior:\n";
    // Optionally loop and print residuals
    for (Dim k = 0; k < g.Nz; ++k)
        for (Dim j = 0; j < g.Ny; ++j)
            for (Dim i = 0; i < g.Nx; ++i)
            {
                Real residual = std::abs(Ax.get(i, j, k) - rhs.get(i, j, k));
                if (residual > 1e-5)
                {
                    std::cout << std::fixed << std::setprecision(6)
                              << "At (i,j,k)=(" << i << "," << j << "," << k << "): "
                              << "A*vector = " << Ax.get(i, j, k)
                              << ", rhs = " << rhs.get(i, j, k)
                              << ", |A*vector - rhs| = " << residual << "\n";
                }
            }
}

// ===============================================================
// 6. Solve and check solution
// ===============================================================
bool solve_and_check(PressureSolver &solver, ScalarVariable &rhs,
                     ScalarVariable &vector, const Grid &g,
                     DimensionsHandlerVector<decltype(define_stride_x(g))> &x_handler)
{

    ScalarVariable computed_sol(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    computed_sol.set_all(0.0);

    solver.solve_pressure<decltype(define_stride_x(g)), 0>(rhs, computed_sol, x_handler);

    ScalarVariable Acomp(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    solver.apply_matrix_operator<0, decltype(define_stride_x(g))>(vector, Acomp, rhs, x_handler);

    Real max_res = 0.0;
    Real tolerance = 1e-2;
    /*

      for (int cc = 0; cc < 3; ++cc)
        for (Dim k = 0; k < g.Nz; ++k)
            for (Dim j = 0; j < g.Ny; ++j)
                for (Dim i = 0; i < g.Nx; ++i)
                {
                    Real residual = std::abs(Acomp.value(cc, i, j, k) - rhs.value(cc, i, j, k));
                    if (residual > tolerance)
                    {
                        printf("Residual error at comp=%d (i,j,k)=(%d,%d,%d): |A*sol - rhs| = %f (tolerance = %f)\n",
                               cc, i, j, k, residual, tolerance);
                        printf("  A*sol = %f, rhs = %f\n", Acomp.value(cc, i, j, k), rhs.value(cc, i, j, k));
                        std::cout << "[FAIL] Residual exceeds tolerance.\n";
                        return false;
                    }
                    max_res = std::max(max_res, residual);
                }
    std::cout << "Max |A*computed_sol - rhs| = " << max_res << "\n";

    */

        for (Dim k = 0; k < g.Nz; ++k)
            for (Dim j = 0; j < g.Ny; ++j)
                for (Dim i = 0; i < g.Nx; ++i)
                    if (std::abs(vector.get(i, j, k) - computed_sol.get(i, j, k)) > tolerance)
                    {
                        printf("Mismatch (i,j,k)=(%d,%d,%d): expected %f, got %f\n",
                               i, j, k, vector.get(i, j, k), computed_sol.get(i, j, k));
                        return false;
                    }

    return true;
}

// ===============================================================
// 7. Main function
// ===============================================================
int main()
{
    constexpr Real two_pi = 2.0 * M_PI;
    Grid g = setup_grid(two_pi, two_pi, two_pi, 20, 20, 20, 0.002f);

    printf("Grid setup: Nx=%d, Ny=%d, Nz=%d, dx=%f, dy=%f, dz=%f, dt=%f\n",
           g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz, g.dt);

    ScalarVariable vector(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    ScalarVariable rhs(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);

    BoundaryFunctions p_boundary;
    std::vector<std::string> exact_bc = {"-sin(t)*2*1e5*cos(x)*sin(y)*(sin(z)-cos(z))"};

    std::vector<std::string> derived_bc = {
        "2*1e5*sin(t)*sin(x)*sin(y)*(sin(z)-cos(z))",
        "-2*1e5*sin(t)*cos(x)*cos(y)*(sin(z)-cos(z))",
        "-2*1e5*sin(t)*cos(x)*sin(y)*(sin(z)+cos(z))"
    };

    BoundaryFunctions p_exact;
    p_exact.set_string_expression(exact_bc);
    p_boundary.set_string_expression(derived_bc);

    initialize_fields(g, vector, rhs, p_boundary, g.dt);
    PressureSolver solver = setup_solver(g, p_boundary, g.dt);

    auto stride_x = define_stride_x(g);
    DimensionsHandlerVector<decltype(stride_x)> x_handler(g.Nx, g.Ny, g.Nz, 0, 1, 2, g.dx, stride_x);

    bool success = solve_and_check(solver, rhs, vector, g, x_handler);

    if (success)
        std::cout << "[PASS] Solver matrix and inversion are consistent.\n";
    else
        std::cout << "[FAIL] Solver test failed.\n";

    return success ? 0 : -1;
}
