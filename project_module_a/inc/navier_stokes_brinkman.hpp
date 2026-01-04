#include <string>
#include <cmath>
#include <functional>
#include <vector>
#include "ScalarVariable.hpp"
#include "VectorVariable.hpp"
#include "Solver.hpp"
#include "ParseInput.hpp"
#include "manufactured_solution_technique.hpp"

class NavierStokesBrinkmann
{
public:
    class Grid
    {

        // Until we get more information for this part we will leave it empty
    public:
        Grid() {}

        void compute_grid() {};
        void split_in_blocks() {};
        void compute_porosity(/*data structure for porosity*/){};

    protected:
    };

public:
    NavierStokesBrinkmann(const Dim Nx, const Dim Ny, const Dim Nz,
                          const Real dt, const Real T,
                          std::string u_boundary_file,
                          std::string p_boundary_file,
                          const Real dx, const Real dy, const Real dz,
                          const Real nu)
        : // ============================
          // GRID, MATERIAL, AND TIME INFO
          // ============================
          grid(),
          dt(dt),
          Nx(Nx),
          Ny(Ny),
          Nz(Nz),
          dx(dx),
          dy(dy),
          dz(dz),
          T(T),

          // ============================
          // SOLVER CLASS
          // ============================
          velocity_solver(Nx, Ny, Nz, dx, dy, dz, dt, gamma_field, u_boundary),
          pressure_solver(Nx, Ny, Nz, dx, dy, dz, dt, p_boundary),

          // ============================
          // PHYSICAL AND MATERIAL FIELDS
          // ============================
          u_0(Nx, Ny, Nz, dx, dy, dz),
          p_0(Nx, Ny, Nz, dx, dy, dz),
          nu(nu),
          k_field(Nx, Ny, Nz, dx, dy, dz),
          gamma_field(Nx, Ny, Nz, dx, dy, dz),
          forcing_field(Nx, Ny, Nz, dx, dy, dz),

          p_boundary(),
          u_boundary(),

          // ============================
          // VECTOR LINEAR SOLVER VARIABLES
          // ============================
          g(Nx, Ny, Nz, dx, dy, dz),
          vector_rhs(Nx, Ny, Nz, dx, dy, dz),
          vector_intermediate_solution(Nx, Ny, Nz, dx, dy, dz),
          xi(Nx, Ny, Nz, dx, dy, dz),
          eta(Nx, Ny, Nz, dx, dy, dz),
          zeta(Nx, Ny, Nz, dx, dy, dz),

          // ============================
          // SCALAR LINEAR SOLVER VARIABLES
          // ============================
          pressure_predictor(Nx, Ny, Nz, dx, dy, dz),
          gradient_pressure_predictor(Nx, Ny, Nz, dx, dy, dz),
          rhs(Nx, Ny, Nz, dx, dy, dz),
          psi(Nx, Ny, Nz, dx, dy, dz),
          phi(Nx, Ny, Nz, dx, dy, dz),
          other_phi(Nx, Ny, Nz, dx, dy, dz),

          // ============================
          // FINAL SOLUTION STORAGE
          // ============================
          velocity_solution(Nx, Ny, Nz, dx, dy, dz),
          pressure_solution(Nx, Ny, Nz, dx, dy, dz)

    {
        // --- Initialize all fields ---

        // 1) PARSER RETRIEVAL
        auto &parser = ParseInput::getInstance();

        // Retrieve functions
        forcing_term_funcion = parser.get_forcing_function_expression();

        forcing_field.set_all(forcing_term_funcion, Real(0.0), true);


        auto k_function = parser.get_k_function_expression();
        k_field.set_all(k_function, Real(0.0));
        

        // // Initialize intermediate fields to zero (CRITICAL FIX)
        // g.set_all(0.0f);

        // vector_rhs.set_all(0.0f);
        // vector_intermediate_solution.set_all(0.0f);
        // xi.set_all(0.0f);
        // eta.set_all(0.0f);
        // zeta.set_all(0.0f);
        // pressure_predictor.set_all(0.0f);
        // rhs.set_all(0.0f);
        // psi.set_all(0.0f);
        // phi.set_all(0.0f);
        // other_phi.set_all(0.0f);

        // // Final solution fields are set to initial conditions in solve()
        // velocity_solution.set_all(0.0f);
        // pressure_solution.set_all(0.0f);

        u_boundary.setParsing(u_boundary_file);
        p_boundary.setParsing(p_boundary_file);
        initialize_gamma_field();
    }
    // initialization methods:
    void initialize_gamma_field();

    Real compute_beta(Dim i, Dim j, Dim k) const;
    Real compute_beta(Dim index) const;

    Real compute_gamma(Dim i, Dim j, Dim k) const;
    Real compute_gamma(Dim index) const;

    // methods inside the iteration:
    void compute_vector_g(Real t);
    void compute_vector_xi();

    void center_pressure(ScalarVariable &pressure_field);

    void compute_rhs_pressure();

    void update_pressure_and_velocity_fields();
    void solve(const ManufacturedSolution &mms);

    /**
     * @brief Computes the L2 relative error between the numerical solution and the MMS exact solution.
     * * @param u_num Numerical velocity field (from Solver)
     * @param p_num Numerical pressure field (from Solver)
     * @param mms   Manufactured Solution instance (provides exact solution and grid spacing)
     * @param t     Time at which to evaluate the error
     * @return std::pair<Real, Real> {Relative Error Velocity, Relative Error Pressure}
     */
    std::pair<Real, Real> compute_L2_errors(
        const VectorVariable &u_num,
        const ScalarVariable &p_num,
        const ManufacturedSolution &mms,
        Real t)
    {

        // 2. Calculate Volume Element
        Real dV = dx * dy * dz;

        Real err_u = 0.0, err_p = 0.0;
        Real norm_u = 0.0, norm_p = 0.0;

        // 3. Loop over grid
        for (Dim k = 1; k < Nz-1; ++k)
        {
            for (Dim j = 1; j < Ny-1; ++j)
            {
                for (Dim i = 1; i < Nx-1; ++i)
                {
                    // Physical Coordinates
                    Real x = i * dx;
                    Real y = j * dy;
                    Real z = k * dz;

                    // --- Exact Solution ---
                    std::vector<Real> u_ex_x = mms.velocity(x+0.5*dx, y, z, t);
                    std::vector<Real> u_ex_y = mms.velocity(x, y+0.5*dy, z, t);
                    std::vector<Real> u_ex_z = mms.velocity(x, y, z+0.5*dz, t);
                    std::vector<Real> u_ex = {u_ex_x[0], u_ex_y[1], u_ex_z[2]};
                    Real p_ex = mms.pressure(x, y, z, t);

                    // --- Numerical Solution ---
                    Real u_num_x = u_num.value(0, i, j, k);
                    Real u_num_y = u_num.value(1, i, j, k);
                    Real u_num_z = u_num.value(2, i, j, k);
                    Real p_val = p_num.get(i, j, k);

                    // --- Velocity Error Accumulation ---
                    Real dux = u_num_x - u_ex[0];
                    Real duy = u_num_y - u_ex[1];
                    Real duz = u_num_z - u_ex[2];

                    err_u += dux * dux + duy * duy + duz * duz;
                    norm_u += u_ex[0] * u_ex[0] + u_ex[1] * u_ex[1] + u_ex[2] * u_ex[2];

                    // --- Pressure Error Accumulation ---
                    err_p += (p_val - p_ex) * (p_val - p_ex);
                    norm_p += p_ex * p_ex;
                }
            }
        }

        // 4. Scale by Volume (L2 Integral approximation)
        err_u = std::sqrt(err_u * dV);
        norm_u = std::sqrt(norm_u * dV);
        err_p = std::sqrt(err_p * dV);
        norm_p = std::sqrt(norm_p * dV);

        // 5. Compute Relative Errors
        // If norm is effectively zero (e.g. at t=0), return the absolute error or 0
        Real rel_err_u = (norm_u > 1e-15) ? err_u / norm_u : 0.0;
        Real rel_err_p = (norm_p > 1e-15) ? err_p / norm_p : 0.0;

        // 6. PRINT DETAILED DIAGNOSTICS (Crucial for debugging)
        std::cout << "------------------------------------------\n";
        std::cout << " ERRORS at t = " << t << "\n";
        std::cout << " Velocity -> Abs: " << err_u << " | Ref Norm: " << norm_u << " | Rel: " << rel_err_u << "\n";
        std::cout << " Pressure -> Abs: " << err_p << " | Ref Norm: " << norm_p << " | Rel: " << rel_err_p << "\n";
        std::cout << "------------------------------------------\n";

        return {rel_err_u, rel_err_p};
    }

    /**
     * @brief Computes the L2 relative error ONLY on the boundary nodes.
     * Useful to verify if Dirichlet Boundary Conditions are being respected/overwritten.
     */
    std::pair<Real, Real> compute_Boundary_L2_errors(
        const VectorVariable &u_num,
        const ScalarVariable &p_num,
        const ManufacturedSolution &mms,
        Real t)
    {
        // 1. Get Grid Dimensions
        Dim Nx = u_num.get_Nx();
        Dim Ny = u_num.get_Ny();
        Dim Nz = u_num.get_Nz();

        // 2. Calculate Volume Element
        // Note: Even on boundary, we treat the node as representing a volume element for consistency
        Real dV = dx * dy * dz;

        Real err_u = 0.0, err_p = 0.0;
        Real norm_u = 0.0, norm_p = 0.0;

        // 3. Loop over grid
        for (Dim k = 0; k < Nz; ++k)
        {
            for (Dim j = 0; j < Ny; ++j)
            {
                for (Dim i = 0; i < Nx; ++i)
                {
                    // // --- FILTER: ONLY PROCESS BOUNDARY NODES ---
                    // bool is_boundary = (i == 0 || i == Nx - 1 ||
                    //                     j == 0 || j == Ny - 1 ||
                    //                     k == 0 || k == Nz - 1);

                    // if (!is_boundary)
                    //     continue; // Skip internal nodes

                    // boundary_node_count++;

                    // Physical Coordinates
                    Real x = i * dx;
                    Real y = j * dy;
                    Real z = k * dz;

                    // --- Exact Solution ---
                    std::vector<Real> u_ex = mms.velocity(x, y, z, t);
                    Real p_ex = mms.pressure(x, y, z, t);

                    // --- Numerical Solution ---
                    Real u_num_x = u_num.value(0, i, j, k);
                    Real u_num_y = u_num.value(1, i, j, k);
                    Real u_num_z = u_num.value(2, i, j, k);
                    Real p_val = p_num.get(i, j, k);

                    // --- Velocity Error Accumulation ---
                    Real dux = u_num_x - u_ex[0];
                    Real duy = u_num_y - u_ex[1];
                    Real duz = u_num_z - u_ex[2];

                    err_u += dux * dux + duy * duy + duz * duz;
                    norm_u += u_ex[0] * u_ex[0] + u_ex[1] * u_ex[1] + u_ex[2] * u_ex[2];

                    // --- Pressure Error Accumulation ---
                    err_p += (p_val - p_ex) * (p_val - p_ex);
                    norm_p += p_ex * p_ex;
                }
            }
        }

        // 4. Scale and Root
        err_u = std::sqrt(err_u * dV);
        norm_u = std::sqrt(norm_u * dV);
        err_p = std::sqrt(err_p * dV);
        norm_p = std::sqrt(norm_p * dV);

        // 5. Compute Relative Errors
        Real rel_err_u = (norm_u > 1e-15) ? err_u / norm_u : 0.0;
        Real rel_err_p = (norm_p > 1e-15) ? err_p / norm_p : 0.0;

        // 6. PRINT DIAGNOSTICS
        // std::cout << "------------------------------------------\n";
        // std::cout << " BOUNDARY ERRORS at t = " << t << " (Nodes: " << boundary_node_count << ")\n";
        // std::cout << " Velocity -> Abs: " << err_u << " | Rel: " << rel_err_u << "\n";
        // std::cout << " Pressure -> Abs: " << err_p << " | Rel: " << rel_err_p << "\n";
        // std::cout << "------------------------------------------\n";

        return {rel_err_u, rel_err_p};
    };

    void write_velocity_vtk(const std::string &filename) const;

    // ============================================================================
    // GRID, MATERIAL, AND TIME INFORMATION
    // ============================================================================
    Grid grid; // Grid geometry and domain decomposition

    Real dt; // Time step

    Dim Nx; // Grid points in x
    Dim Ny; // Grid points in y
    Dim Nz; // Grid points in z

    Real dx = 1.0f; // Grid spacing in x
    Real dy = 1.0f; // Grid spacing in y
    Real dz = 1.0f; // Grid spacing in z

    Real T;

    // ============================================================================
    // SOLVER CLASS
    // ============================================================================
    VelocitySolver velocity_solver;
    PressureSolver pressure_solver;

    // ============================================================================
    // PHYSICAL AND MATERIAL FIELDS
    // ============================================================================
    VectorVariable u_0;                                                        // Velocity field
    ScalarVariable p_0;                                                        // Pressure field
    VectorVariable forcing_field; // Forcing term (can vary in space)
    Real nu;                                                                   // Kinematic viscosity (can vary in space)
    ScalarVariable k_field;                                                    // Brinkman permeability or resistance term
    ScalarVariable gamma_field;                                                // Gamma field for Brinkman term
    BoundaryFunctions p_boundary;                                              // Boundary condition for pressure
    BoundaryFunctions u_boundary;                                              // Boundary condition for velocity
    BoundaryFunctions forcing_term_funcion;

    // ============================================================================
    // VECTOR LINEAR SOLVER VARIABLES (MOMENTUM EQUATION)
    //    Used to solve the three components of momentum
    // ============================================================================
    VectorVariable g;
    VectorVariable vector_rhs;                   // RHS of the momentum equation
    VectorVariable vector_intermediate_solution; // solution of linear equation, it's a delta between
                                                 // the previous timestamp variable and the new one
    VectorVariable xi;                           // x-direction solve intermediate
    VectorVariable eta;                          // y-direction solve intermediate
    VectorVariable zeta;                         // z-direction solve intermediate

    // ============================================================================
    // SCALAR LINEAR SOLVER VARIABLES (PRESSURE EQUATION AND OTHER SCALARS)
    // ============================================================================
    ScalarVariable pressure_predictor;          // Pressure predictor
    VectorVariable gradient_pressure_predictor; // ∇p correction term
    ScalarVariable rhs;                         // RHS of scalar Poisson equation
    ScalarVariable psi;                         // Auxiliary scalar (potential or correction)
    ScalarVariable phi;                         // Pressure correction
    ScalarVariable other_phi;                   // Additional scalar field for iterative updates

    // ============================================================================
    // FINAL SOLUTION STORAGE
    // ============================================================================
    VectorVariable velocity_solution; // Final converged velocity
    ScalarVariable pressure_solution; // Final converged pressure

    std::vector<VectorVariable> velocity_time_series;
    std::vector<ScalarVariable> pressure_time_series;
};
