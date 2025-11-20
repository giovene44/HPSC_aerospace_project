#include <string>
#include <cmath>
#include <functional>
#include <vector>
#include "ScalarVariable.hpp"
#include "VectorVariable.hpp"
#include "Solver.hpp"
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
                          std::function<std::vector<Real>(Real, Real, Real, Real)> forcing_func,
                          std::function<Real(Real, Real, Real)> k_func,
                          std::string u_boundary_file,
                          std::string p_boundary_file,
                          const Real dx, const Real dy, const Real dz,
                          Real Re = 100.0f)
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
          forcing_function(forcing_func),
          k_function(k_func),
          Re(Re),
          k_field(Nx, Ny, Nz, dx, dy, dz),
          gamma_field(Nx, Ny, Nz, dx, dy, dz),

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
        nu = 1.0f / Re;
        initialize_k_field();
        initialize_gamma_field();
    }
    // initialization methods:
    void initialize_gamma_field();
    void initialize_k_field();

    Real compute_beta(Dim i, Dim j, Dim k) const;
    Real compute_beta(Dim index) const;

    Real compute_gamma(Dim i, Dim j, Dim k) const;
    Real compute_gamma(Dim index) const;

    // methods inside the iteration:
    void compute_vector_g(Real t);
    void compute_vector_xi();

    void compute_rhs_pressure();

    void update_pressure_and_velocity_fields();
    void solve();

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
    std::function<std::vector<Real>(Real, Real, Real, Real)> forcing_function; // Forcing term (can vary in space)
    std::function<Real(Real, Real, Real)> k_function;                          // Forcing term (can vary in space)
    Real Re;
    Real nu;                      // Kinematic viscosity (can vary in space)
    ScalarVariable k_field;       // Brinkman permeability or resistance term
    ScalarVariable gamma_field;   // Gamma field for Brinkman term
    BoundaryFunctions p_boundary; // Boundary condition for pressure
    BoundaryFunctions u_boundary; // Boundary condition for velocity

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
};
