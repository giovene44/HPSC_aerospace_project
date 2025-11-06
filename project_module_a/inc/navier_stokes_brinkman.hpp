#include <string>
#include <cmath>
#include "ScalarVariable.hpp"
#include "VectorVariable.hpp"
#include "DimensionsHandler.hpp"

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

public:
    NavierStokesBrinkmann(const Dim Nx, const Dim Ny, const Dim Nz, const float dt, const Real dx = 1.0f, const Real dy = 1.0f, const Real dz = 1.0f)
        : Nx(Nx), Ny(Ny), Nz(Nz), dt(dt), dx(dx), dy(dy), dz(dz),
          u_0(Nx, Ny, Nz, dx, dy, dz),
          p_0(Nx, Ny, Nz, dx, dy, dz),
          f(Nx, Ny, Nz, dx, dy, dz),
          nu(Nx, Ny, Nz, dx, dy, dz),
          k_field(Nx, Ny, Nz, dx, dy, dz),
          gamma_field(Nx, Ny, Nz, dx, dy, dz),
          g(Nx, Ny, Nz, dx, dy, dz),
          vector_gamma_D_term(Nx, Ny, Nz, dx, dy, dz),
          vector_rhs(Nx, Ny, Nz, dx, dy, dz),
          u_1(Nx, Ny, Nz, dx, dy, dz),
          xi(Nx, Ny, Nz, dx, dy, dz),
          eta_0(Nx, Ny, Nz, dx, dy, dz),
          eta_1(Nx, Ny, Nz, dx, dy, dz),
          zeta_0(Nx, Ny, Nz, dx, dy, dz),
          zeta_1(Nx, Ny, Nz, dx, dy, dz),
          gradient_pressure(Nx, Ny, Nz, dx, dy, dz),
          velocity_solution(Nx, Ny, Nz, dx, dy, dz),
          rhs(Nx, Ny, Nz, dx, dy, dz),
          psi(Nx, Ny, Nz, dx, dy, dz),
          phi(Nx, Ny, Nz, dx, dy, dz),
          other_phi(Nx, Ny, Nz, dx, dy, dz),
          sol_linear_system(Nx, Ny, Nz, dx, dy, dz),
          a(Nx, Ny, Nz, dx, dy, dz),
          b(Nx, Ny, Nz, dx, dy, dz),
          c(Nx, Ny, Nz, dx, dy, dz),
          pressure_solution(Nx, Ny, Nz, dx, dy, dz)
    {
        // constructor body (leave empty or add initialization code here)
    }
    // initialization methods:
    void initialize_gamma_field();

    // output should be passed by reference and should be allocated in the costructor of the class "!!!"    IMPORTANT
    // Function declarations only; implementations moved to the .cpp file
    void compute_vector_difference(VectorVariable &output, const VectorVariable &v1, const VectorVariable &v2);

    void thomas_algorithm(const std::vector<float> &a, const std::vector<float> &b, const std::vector<float> &c, const std::vector<float> &rhs, std::vector<float> &x)
    {
        int n = rhs.size();
        std::vector<float> c_prime(c.size(), 0.0);
        std::vector<float> rhs_prime(n, 0.0);

        c_prime[0] = c[0] / b[0];
        rhs_prime[0] = rhs[0] / b[0];

        for (int i = 1; i < n; ++i)
        {
            float m = 1.0 / (b[i] - a[i] * c_prime[i - 1]);
            c_prime[i] = c[i] * m;
            rhs_prime[i] = (rhs[i] - a[i] * rhs_prime[i - 1]) * m;
        }

        x[n - 1] = rhs_prime[n - 1];

        for (int i = n - 2; i >= 0; --i)
        {
            x[i] = rhs_prime[i] - c_prime[i] * x[i + 1];
        }
    }
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
    void compute_scalar_rhs_pressure_1();
    void impose_Neumann_bc_scalar(int direction, int side, float neumann_boundary_value);

    template<typename StrideFunction>
    void NavierStokesBrinkmann::block_solver(const ScalarVariable &rhs, ScalarVariable &solution, const DimensionsHandler<StrideFunction> &dim_hand);
    template<typename StrideFunction>
    void NavierStokesBrinkmann::block_solver(const ScalarVariable &rhs, const ScalarVariable &gamma, ScalarVariable &solution, const DimensionsHandler<StrideFunction> &dim_hand);

    // TODO:
    /*
    - compute_diagonal_terms a,b,c
    - boundary conditions
    - solve scalar linear system
    - solve vector linear system
    - include DimensionHandler (stride functions has to be declared in the solve functions)
    */

    // ============================================================================
    // GRID, MATERIAL, AND TIME INFORMATION
    // ============================================================================
    Grid grid; // Grid geometry and domain decomposition

    float dt; // Time step

    Dim Nx; // Grid points in x
    Dim Ny; // Grid points in y
    Dim Nz; // Grid points in z

    Real dx = 1.0f; // Grid spacing in x
    Real dy = 1.0f; // Grid spacing in y
    Real dz = 1.0f; // Grid spacing in z

    // ============================================================================
    // PHYSICAL AND MATERIAL FIELDS
    // ============================================================================
    VectorVariable u_0;          // Velocity field
    ScalarVariable p_0;         // Pressure field
    VectorVariable f;            // Forcing term (can vary in space)
    ScalarVariable nu;          // Kinematic viscosity (can vary in space)
    ScalarVariable k_field;     // Brinkman permeability or resistance term
    ScalarVariable gamma_field; // Gamma field for Brinkman term

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

    // ============================================================================
    // SCALAR LINEAR SOLVER VARIABLES (PRESSURE EQUATION AND OTHER SCALARS)
    // ============================================================================
    ScalarVariable rhs;       // RHS of scalar Poisson equation
    ScalarVariable psi;       // Auxiliary scalar (potential or correction)
    ScalarVariable phi;       // Pressure correction
    ScalarVariable other_phi; // Additional scalar field for iterative updates

    ScalarVariable sol_linear_system; // Temporary solution of scalar linear system
    ScalarVariable a;                 // Tridiagonal coefficient a (lower diag)
    ScalarVariable b;                 // Tridiagonal coefficient b (main diag)
    ScalarVariable c;                 // Tridiagonal coefficient c (upper diag)
    // ============================================================================
    // FINAL SOLUTION STORAGE
    // ============================================================================
    VectorVariable velocity_solution;  // Final converged velocity
    ScalarVariable pressure_solution; // Final converged pressure
};
