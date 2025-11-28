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
            /*
            vector.set(1, 0, j, k) = u_boundary.value<1>(0.0, 0.5 * g.dy + j * g.dy, k * g.dz, t);
            Real val = vector.value(1, g.Nx - 1, j, k);
            Real gamma_val = gamma_field.get(g.Nx - 1, j, k);
            Real sd = vector.second_derivative(1, 0, g.Nx - 1, j, k);
            Real rhs_val = val - gamma_val * sd;
            auto a = -gamma_val / (g.dx * g.dx);
            auto b = 1.0 + 2.0 * gamma_val / (g.dx * g.dx);
            auto c = -gamma_val / (g.dx * g.dx);
            vector.set(1, g.Nx - 1, j, k) = ((rhs_val - 2 * c * u_boundary.value<1>(Lx, 0.5 * g.dy + j * g.dy, k * g.dz, t)) - a * vector.value(1, g.Nx - 2, j, k)) / (b - c);
            */

            vector.set(2, 0, j, k) = u_boundary.value<2>(0.0, j * g.dy, 0.5 * g.dz + k * g.dz, t);
            Real val2 = vector.value(2, g.Nx - 1, j, k);
            Real gamma_val2 = gamma_field.get(g.Nx - 1, j, k);
            Real sd2 = vector.second_derivative(2, 0, g.Nx - 1, j, k);
            Real rhs_val2 = val2 - gamma_val2 * sd2;
            auto a2 = -gamma_val2 / (g.dx * g.dx);
            auto b2 = 1.0 + 2.0 * gamma_val2 / (g.dx * g.dx);
            auto c2 = -gamma_val2 / (g.dx * g.dx);
            vector.set(2, g.Nx - 1, j, k) = ((rhs_val2 - 2 * c2 * u_boundary.value<2>(Lx, j * g.dy, 0.5 * g.dz + k * g.dz, t)) - a2 * vector.value(2, g.Nx - 2, j, k)) / (b2 - c2);
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
                       BoundaryFunctions &u_boundary,
                       Real t)
{

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

                    vector.set(comp, i, j, k) = sin(x) * sin(t);
                }

    // Apply known faces
    // apply_known_faces(vector, u_boundary, g, gamma_field, 0.0);

    // Build RHS = (I - gamma * Dxx) * vector
    for (int comp = 0; comp < 3; ++comp)
        for (Dim k = 0; k < g.Nz; ++k)
            for (Dim j = 0; j < g.Ny; ++j)
                for (Dim i = 0; i < g.Nx; ++i)
                {
                    Real val = vector.value(comp, i, j, k);
                    Real gamma_val = gamma_field.get(i, j, k);
                    Real sd = vector.second_derivative(comp, 0, i, j, k);
                    Real rhs_val = val - gamma_val * sd;

                    // Apply boundary BCs on RHS

                    if (i == 0 || i == g.Nx - 1 || j == 0 || j == g.Ny - 1 || k == 0 || k == g.Nz - 1)
                    {
                        Real x_coord, y_coord, z_coord;
                        if (comp == 0)
                        {
                            x_coord = (i + 0.5) * g.dx;
                            y_coord = j * g.dy;
                            z_coord = k * g.dz;
                        }
                        else if (comp == 1)
                        {
                            x_coord = i * g.dx;
                            y_coord = (j + 0.5) * g.dy;
                            z_coord = k * g.dz;
                        }
                        else
                        {
                            x_coord = i * g.dx;
                            y_coord = j * g.dy;
                            z_coord = (k + 0.5) * g.dz;
                        }
                        if (comp == 0)
                        {
                            rhs_val = u_boundary.value<0>(x_coord, y_coord, z_coord, t);
                            vector.set(comp, i, j, k) = rhs_val;
                        }

                        else if (comp == 1)
                        {
                            rhs_val = u_boundary.value<1>(x_coord, y_coord, z_coord, t);
                            vector.set(comp, i, j, k) = rhs_val;
                        }
                        else
                        {
                            rhs_val = u_boundary.value<2>(x_coord, y_coord, z_coord, t);
                            vector.set(comp, i, j, k) = rhs_val;
                        }
                    }
                    rhs.set(comp, i, j, k) = rhs_val;
                }
}

// ===============================================================
// 4. Setup Solver & Strides
// ===============================================================
VelocitySolver setup_solver(const Grid &g, ScalarVariable &gamma_field, BoundaryFunctions &u_boundary, Real t)
{
    VelocitySolver solver(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz, t, gamma_field, u_boundary);
    solver.gamma_field = gamma_field;
    solver.u_boundary = u_boundary;
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
void check_matrix_operator(VelocitySolver &solver, const VectorVariable &vector,
                           VectorVariable &rhs, const Grid &g,
                           DimensionsHandlerVector<decltype(define_stride_x(g))> &x_handler)
{
    VectorVariable Ax(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    solver.apply_matrix_operator<0, decltype(define_stride_x(g))>(vector, Ax, rhs, x_handler);
    std::cout << "Matrix operator diagnostic on boundaries and interior:\n";
    // Optionally loop and print residuals
}

// ===============================================================
// 6. Solve and check solution
// ===============================================================
bool solve_and_check(VelocitySolver &solver, VectorVariable &rhs_1,
                     VectorVariable &vector_1, VectorVariable &rhs_2,
                     VectorVariable &vector_2, const Grid &g,
                     DimensionsHandlerVector<decltype(define_stride_x(g))> &x_handler)
{
    VectorVariable rhs_delta(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    VectorVariable vector_delta(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);

    rhs_delta = rhs_2 - rhs_1;
    vector_delta = vector_2 - vector_1;

    VectorVariable computed_sol(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    computed_sol.set_all(0.0);

    solver.solve<decltype(define_stride_x(g)), 0>(rhs_delta, computed_sol, x_handler);

    VectorVariable Acomp(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    solver.apply_matrix_operator<0, decltype(define_stride_x(g))>(vector_delta, Acomp, rhs_delta, x_handler);

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
                    if (std::abs(vector_delta.value(comp, i, j, k) - computed_sol.value(comp, i, j, k)) > tolerance)
                    {
                        printf("Mismatch comp=%d (i,j,k)=(%d,%d,%d): expected %f, got %f\n",
                               comp, i, j, k, vector_delta.value(comp, i, j, k), computed_sol.value(comp, i, j, k));
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
    Grid g = setup_grid(two_pi, two_pi, two_pi, 100, 100, 100, 0.002f);

    printf("Grid setup: Nx=%d, Ny=%d, Nz=%d, dx=%f, dy=%f, dz=%f, dt=%f\n",
           g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz, g.dt);

    ScalarVariable gamma_field(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    gamma_field.set_all(0.1f);
    VectorVariable vector_1(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    VectorVariable rhs_1(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    VectorVariable vector_2(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    VectorVariable rhs_2(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);

    BoundaryFunctions u_boundary;
    std::vector<std::string> sin_bc = {"sin(x)*sin(t)", "sin(x)*sin(t)", "sin(x)*sin(t)"};
    u_boundary.set_string_expression(sin_bc);

    initialize_fields(g, gamma_field, vector_1, rhs_1, u_boundary, g.dt);
    initialize_fields(g, gamma_field, vector_2, rhs_2, u_boundary, g.dt + g.dt);

    VelocitySolver solver = setup_solver(g, gamma_field, u_boundary, g.dt + g.dt);

    auto stride_x = define_stride_x(g);
    DimensionsHandlerVector<decltype(stride_x)> x_handler(g.Nx, g.Ny, g.Nz, 0, 1, 2, g.dx, stride_x);

    bool success = solve_and_check(solver, rhs_1, vector_1, rhs_2, vector_2, g, x_handler);

    if (success)
        std::cout << "[PASS] Solver matrix and inversion are consistent.\n";
    else
        std::cout << "[FAIL] Solver test failed.\n";

    return success ? 0 : -1;
}
