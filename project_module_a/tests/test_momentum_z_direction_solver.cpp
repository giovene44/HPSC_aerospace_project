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
// 2. Apply known Dirichlet faces on vector [NEVER CALLED]
// ===============================================================
void apply_known_faces(VectorVariable &vector,
                       BoundaryFunctions &u_boundary,
                       const Grid &g,
                       ScalarVariable &gamma_field,
                       Real t = 0.0)
{
    Real Lx = g.dx * (g.Nx - 0.5);
    Real Ly = g.dy * (g.Ny - 0.5);
    Real Lz = g.dz * (g.Nz - 0.5);

    // X-direction faces
    for (Dim j = 0; j < g.Ny; ++j)
        for (Dim k = 0; k < g.Nz; ++k)
        {
            vector.set(0, 0, j, k) = u_boundary.value<0>(0.0, j * g.dy, k * g.dz, t);
            vector.set(0, g.Nx - 1, j, k) = u_boundary.value<0>(Lx, j * g.dy, k * g.dz, t);

            vector.set(1, 0, j, k) = u_boundary.value<1>(0.0, 0.5 * g.dy + j * g.dy, k * g.dz, t);
            vector.set(1, g.Nx - 1, j, k) = u_boundary.value<1>(Lx, 0.5 * g.dy + j * g.dy, k * g.dz, t);

            vector.set(2, 0, j, k) = u_boundary.value<2>(0.0, j * g.dy, 0.5 * g.dz + k * g.dz, t);
            vector.set(2, g.Nx - 1, j, k) = u_boundary.value<2>(Lx, j * g.dy, 0.5 * g.dz + k * g.dz, t);
        }

    // Y-direction faces
    for (Dim i = 0; i < g.Nx; ++i)
        for (Dim k = 0; k < g.Nz; ++k)
        {
            vector.set(1, i, 0, k) = u_boundary.value<1>(i * g.dx, 0.0, k * g.dz, t);
            vector.set(1, i, g.Ny - 1, k) = u_boundary.value<1>(i * g.dx, Ly, k * g.dz, t);

            vector.set(0, i, 0, k) = u_boundary.value<0>(0.5 * g.dx + i * g.dx, 0.0, k * g.dz, t);
            vector.set(0, i, g.Ny - 1, k) = u_boundary.value<0>(0.5 * g.dx + i * g.dx, Ly, k * g.dz, t);

            vector.set(2, i, 0, k) = u_boundary.value<2>(i * g.dx, 0.0, 0.5 * g.dz + k * g.dz, t);
            vector.set(2, i, g.Ny - 1, k) = u_boundary.value<2>(i * g.dx, Ly, 0.5 * g.dz + k * g.dz, t);
        }

    // Z-direction faces
    for (Dim i = 0; i < g.Nx; ++i)
        for (Dim j = 0; j < g.Ny; ++j)
        {
            vector.set(2, i, j, 0) = u_boundary.value<2>(i * g.dx, j * g.dy, 0.0, t);
            vector.set(2, i, j, g.Nz - 1) = u_boundary.value<2>(i * g.dx, j * g.dy, Lz, t);

            vector.set(0, i, j, 0) = u_boundary.value<0>(0.5 * g.dx + i * g.dx, j * g.dy, 0.0, t);
            vector.set(0, i, j, g.Nz - 1) = u_boundary.value<0>(0.5 * g.dx + i * g.dx, j * g.dy, Lz, t);

            vector.set(1, i, j, 0) = u_boundary.value<1>(i * g.dx, 0.5 * g.dy + j * g.dy, 0.0, t);
            vector.set(1, i, j, g.Nz - 1) = u_boundary.value<1>(i * g.dx, 0.5 * g.dy + j * g.dy, Lz, t);
        }
}

// ===============================================================
// 3. Initialize Fields (interior + RHS)
// ===============================================================
void initialize_fields(const Grid &g,
                       ScalarVariable &gamma_field,
                       VectorVariable &vector,
                       VectorVariable &rhs,
                       BoundaryFunctions &u_boundary)
{
    gamma_field.set_all(0.1f);

    // Initialize interior
    for (int comp = 0; comp < 3; ++comp)
        for (int k = 0; k < g.Nz; ++k)
            for (int j = 0; j < g.Ny; ++j)
                for (int i = 0; i < g.Nx; ++i)
                {
                    Real x, y, z;
                    if (comp == 0)
                    {
                        x = (i + 0.5) * g.dx;
                        y = j * g.dy;
                        z = k * g.dz;
                    }
                    else if (comp == 1)
                    {
                        x = i * g.dx;
                        y = (j + 0.5) * g.dy;
                        z = k * g.dz;
                    }
                    else
                    {
                        x = i * g.dx;
                        y = j * g.dy;
                        z = (k + 0.5) * g.dz;
                    }

                    vector.set(comp, i, j, k) = sin(z);
                }

    // Apply known faces
    // apply_known_faces(vector, u_boundary, g, gamma_field, 0.0);

    // Build RHS = (I - gamma * Dzz) * vector
    for (int comp = 0; comp < 3; ++comp)
        for (Dim k = 0; k < g.Nz; ++k)
            for (Dim j = 0; j < g.Ny; ++j)
                for (Dim i = 0; i < g.Nx; ++i)
                {
                    Real val = vector.value(comp, i, j, k);
                    Real gamma_val = gamma_field.get(i, j, k);
                    Real sd = vector.second_derivative(comp, 2, i, j, k);
                    Real rhs_val = val - gamma_val * sd;

                    // Apply boundary BCs on RHS

                    if (i == 0 || i == g.Nx - 1 || j == 0 || j == g.Ny - 1 || k == 0 || k == g.Nz - 1)
                    {
                        Real x_coord = (comp == 0) ? ((i == 0) ? 0.0 : g.dx * (g.Nx - 0.5)) : (i * g.dx + ((comp == 0) ? 0.5 * g.dx : 0.0));
                        Real y_coord = (comp == 1) ? ((j == 0) ? 0.0 : g.dy * (g.Ny - 0.5)) : (j * g.dy + ((comp == 1) ? 0.5 * g.dy : 0.0));
                        Real z_coord = (comp == 2) ? ((k == 0) ? 0.0 : g.dz * (g.Nz - 0.5)) : (k * g.dz + ((comp == 2) ? 0.5 * g.dz : 0.0));

                        if (comp == 0)
                        {
                            rhs_val = u_boundary.value<0>(x_coord, y_coord, z_coord, 0.0);
                            vector.set(comp, i, j, k) = rhs_val;
                        }

                        else if (comp == 1)
                        {
                            rhs_val = u_boundary.value<1>(x_coord, y_coord, z_coord, 0.0);
                            vector.set(comp, i, j, k) = rhs_val;
                        }
                        else
                        {
                            rhs_val = u_boundary.value<2>(x_coord, y_coord, z_coord, 0.0);
                            vector.set(comp, i, j, k) = rhs_val;
                        }
                    }
                    rhs.set(comp, i, j, k) = rhs_val;
                }
}

// ===============================================================
// 4. Setup Solver & Strides
// ===============================================================
VelocitySolver setup_solver(const Grid &g, ScalarVariable &gamma_field, BoundaryFunctions &u_boundary)
{
    VelocitySolver solver(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz, g.dt, gamma_field, u_boundary);
    return solver;
}

auto define_stride_z(const Grid &g)
{
    return [=](Dim i, Dim j)
    { return i + j * g.Nx; };
}

// ===============================================================
// 5. Diagnostics: Check matrix operator & residual
// ===============================================================
void check_matrix_operator(VelocitySolver &solver, const VectorVariable &vector,
                           VectorVariable &rhs, const Grid &g,
                           DimensionsHandlerVector<decltype(define_stride_z(g))> &z_handler)
{
    VectorVariable Ax(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    solver.apply_matrix_operator<2, decltype(define_stride_z(g))>(vector, Ax, rhs, z_handler);
    std::cout << "Matrix operator diagnostic on boundaries and interior:\n";
    // Optionally loop and print residuals
}

// ===============================================================
// 6. Solve and check solution
// ===============================================================
bool solve_and_check(VelocitySolver &solver, VectorVariable &rhs,
                     const VectorVariable &vector, const Grid &g,
                     DimensionsHandlerVector<decltype(define_stride_z(g))> &z_handler)
{
    VectorVariable computed_sol(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    computed_sol.set_all(0.0);

    solver.solve<decltype(define_stride_z(g)), 2>(rhs, computed_sol, z_handler);

    VectorVariable Acomp(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    solver.apply_matrix_operator<2, decltype(define_stride_z(g))>(vector, Acomp, rhs, z_handler);

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

    for (int comp = 0; comp < 3; ++comp)
        for (Dim k = 0; k < g.Nz; ++k)
            for (Dim j = 0; j < g.Ny; ++j)
                for (Dim i = 0; i < g.Nx; ++i)
                    if (std::abs(vector.value(comp, i, j, k) - computed_sol.value(comp, i, j, k)) > tolerance)
                    {
                        printf("Mismatch comp=%d (i,j,k)=(%d,%d,%d): expected %f, got %f\n",
                               comp, i, j, k, vector.value(comp, i, j, k), computed_sol.value(comp, i, j, k));
                        return false;
                    }

    return true;
}

// ===============================================================
// 7. Main function
// ===============================================================
int main()
{
    const Real two_pi = 2.0 * 3.141592653589793;
    Grid g = setup_grid(two_pi, two_pi, two_pi, 500, 500, 500, 0.01);

    printf("Grid setup: Nx=%d, Ny=%d, Nz=%d, dx=%f, dy=%f, dz=%f, dt=%f\n",
           g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz, g.dt);

    ScalarVariable gamma_field(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    VectorVariable vector(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    VectorVariable rhs(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);

    BoundaryFunctions u_boundary;
    std::vector<std::string> sin_bc = {"0", "0", "0"};
    u_boundary.set_string_expression(sin_bc);

    initialize_fields(g, gamma_field, vector, rhs, u_boundary);

    VelocitySolver solver = setup_solver(g, gamma_field, u_boundary);

    auto stride_z = define_stride_z(g);
    DimensionsHandlerVector<decltype(stride_z)> z_handler(g.Nx, g.Ny, g.Nz, 2, 0, 1, g.dz, stride_z);

    check_matrix_operator(solver, vector, rhs, g, z_handler);

    bool success = solve_and_check(solver, rhs, vector, g, z_handler);

    if (success)
        std::cout << "[PASS] Solver matrix and inversion are consistent.\n";
    else
        std::cout << "[FAIL] Solver test failed.\n";

    return success ? 0 : -1;
}
