#include <string>
#include <cmath>
#include "ScalarVariables.hpp"
#include "VectorVariable.hpp"

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

    NavierStokesBrinkmann();
    void solve();

protected:
    // initialization methods:
    void initialize_gamma_field();

    // output should be passed by reference and should be allocated in the costructor of the class "!!!"    IMPORTANT
    // Function declarations only; implementations moved to the .cpp file
    void compute_vector_difference(VectorVariable &output, const VectorVariable &v1, const VectorVariable &v2);

    Real compute_beta(Dim i, Dim j, Dim k) const;
    Real compute_beta(Dim index) const;

    Real compute_gamma(Dim i, Dim j, Dim k) const;
    Real compute_gamma(Dim index) const;

    // methods inside the iteration:
    void compute_vector_g();
    void compute_vector_xi();
    void compute_gradient_pressure_field();
    void compute_vector_gamma_D_term(int direction);
    void compute_vector_rhs(const VectorVariable &vector1, const VectorVariable &vector2);

    
    void block_solver(int derivation_direction, const ScalarVariables &rhs, const ScalarVariables &gamma, ScalarVariables &solution);
    void block_solver(int derivation_direction, const ScalarVariables &rhs, ScalarVariables &solution);
    /*
    // setup methods:
    void setup(); // read grid, setup initial values, assemble K, gamma, initial conditions
    void solve_linear_systems(auto &rhs, auto &a, auto &b, auto &c, auto &output);
    void update_variables();
    void solve_momentum();
    void solve_pressure();
    void output_results(int timestep) const;
    */

    // ============================================================================
    // GRID, MATERIAL, AND TIME INFORMATION
    // ============================================================================
    Grid grid; // Grid geometry and domain decomposition

    float dt; // Time step

    Dim Nx; // Grid points in x
    Dim Ny; // Grid points in y
    Dim Nz; // Grid points in z

    // ============================================================================
    // PHYSICAL AND MATERIAL FIELDS
    // ============================================================================
    VectorVariable u_0;          // Velocity field
    ScalarVariables p_0;         // Pressure field
    VectorVariable f;            // Forcing term (can vary in space)
    VectorVariable nu;           // Kinematic viscosity (can vary in space)
    ScalarVariables k_field;     // Brinkman permeability or resistance term
    ScalarVariables gamma_field; // Gamma field for Brinkman term

    // ============================================================================
    // VECTOR LINEAR SOLVER VARIABLES (MOMENTUM EQUATION)
    //    Used to solve the three components of momentum
    // ============================================================================
    VectorVariable g;
    VectorVariable vector_gamma_D_term; // γ·D term in the momentum equation
    VectorVariable vector_rhs;          // RHS of the momentum equation
    VectorVariable u_1;                 // Velocity field
    VectorVariable xi;                  // x-direction solve intermediate
    VectorVariable eta_0;               // y-direction solve intermediate
    VectorVariable eta_1;               // y-direction solve intermediate
    VectorVariable zeta_0;              // z-direction solve intermediate
    VectorVariable zeta_1;              // z-direction solve intermediate
    VectorVariable gradient_pressure;   // ∇p correction term
    VectorVariable velocity_solution;   // Final velocity solution

    // ============================================================================
    // SCALAR LINEAR SOLVER VARIABLES (PRESSURE EQUATION AND OTHER SCALARS)
    // ============================================================================
    ScalarVariables rhs;       // RHS of scalar Poisson equation
    ScalarVariables psi;       // Auxiliary scalar (potential or correction)
    ScalarVariables phi;       // Pressure correction
    ScalarVariables other_phi; // Additional scalar field for iterative updates

    ScalarVariables sol_linear_system;  // Temporary solution of scalar linear system
    ScalarVariables a;                  // Tridiagonal coefficient a (lower diag)
    ScalarVariables b;                  // Tridiagonal coefficient b (main diag)
    ScalarVariables c;                  // Tridiagonal coefficient c (upper diag)
    ScalarVariables pressure_predictor; // Intermediate pressure estimate

    // ============================================================================
    // FINAL SOLUTION STORAGE
    // ============================================================================
    VectorVariable velocity_solution;  // Final converged velocity
    ScalarVariables pressure_solution; // Final converged pressure
};