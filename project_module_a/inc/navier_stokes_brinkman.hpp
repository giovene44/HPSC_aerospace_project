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
    NavierStokesBrinkmann(const Dim Nx, const Dim Ny, const Dim Nz, const Real dt, const Real T, const Real dx = 1.0f, const Real dy = 1.0f, const Real dz = 1.0f)
        : Nx(Nx), Ny(Ny), Nz(Nz), dt(dt), T(T), dx(dx), dy(dy), dz(dz),
          u_0(Nx, Ny, Nz, dx, dy, dz),
          u_boundary(Nx, Ny, Nz, dx, dy, dz),
          p_boundary(Nx, Ny, Nz, dx, dy, dz),
          f(Nx, Ny, Nz, dx, dy, dz),
          nu(Nx, Ny, Nz, dx, dy, dz),
          k_field(Nx, Ny, Nz, dx, dy, dz),
          gamma_field(Nx, Ny, Nz, dx, dy, dz),
          g(Nx, Ny, Nz, dx, dy, dz),
          vector_gamma_D_term(Nx, Ny, Nz, dx, dy, dz),
          vector_rhs(Nx, Ny, Nz, dx, dy, dz),
          delta_u(Nx, Ny, Nz, dx, dy, dz),
          xi(Nx, Ny, Nz, dx, dy, dz),
          eta(Nx, Ny, Nz, dx, dy, dz),
          delta_eta(Nx, Ny, Nz, dx, dy, dz),
          zeta(Nx, Ny, Nz, dx, dy, dz),
          delta_zeta(Nx, Ny, Nz, dx, dy, dz),
          gradient_pressure(Nx, Ny, Nz, dx, dy, dz),
          pressure_predictor(Nx, Ny, Nz, dx, dy, dz),
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

    void thomas_algorithm(const std::vector<Real> &a, const std::vector<Real> &b, const std::vector<Real> &c, const std::vector<Real> &rhs, std::vector<Real> &x)
    {
        int n = rhs.size();
        std::vector<Real> c_prime(c.size(), 0.0);
        std::vector<Real> rhs_prime(n, 0.0);

        c_prime[0] = c[0] / b[0];
        rhs_prime[0] = rhs[0] / b[0];

        for (int i = 1; i < n; ++i)
        {
            Real m = 1.0 / (b[i] - a[i] * c_prime[i - 1]);
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
    void compute_scalar_rhs_pressure();

    template<typename StrideFunction>
    void impose_Dirichlet(VectorVariable &rhs, const DimensionsHandlerVector<StrideFunction> &dim_hand);

    template<typename StrideFunction>
    void impose_Neumann(ScalarVariable &rhs, const DimensionsHandlerScalar<StrideFunction> &dim_hand);

    template<typename StrideFunction>
    void NavierStokesBrinkmann::block_solver(const ScalarVariable &rhs, ScalarVariable &solution, const DimensionsHandlerScalar<StrideFunction> &dim_hand);
    template<typename StrideFunction>
    void NavierStokesBrinkmann::block_solver(const VectorVariable &rhs, const ScalarVariable &gamma, VectorVariable &solution, const DimensionsHandlerVector<StrideFunction> &dim_hand);

    // TODO:
    /*
    - boundary conditions
    - solve scalar linear system
    - solve vector linear system
    - include DimensionHandler (stride functions has to be declared in the solve functions)
    */

    // ============================================================================
    // GRID, MATERIAL, AND TIME INFORMATION
    // ============================================================================
    Grid grid; // Grid geometry and domain decomposition

    Real dt; // Time step
    Real T;  // Total simulation time

    Dim Nx; // Grid points in x
    Dim Ny; // Grid points in y
    Dim Nz; // Grid points in z

    Real dx = 1.0f; // Grid spacing in x
    Real dy = 1.0f; // Grid spacing in y
    Real dz = 1.0f; // Grid spacing in z

    // ============================================================================
    // PHYSICAL AND MATERIAL FIELDS
    // ============================================================================
    VectorVariable u_0;         // Inital condition for velocity
    ScalarVariable p_boundary;  // Neumann condition for pressure
    VectorVariable u_boundary;  // Dirichlet Boundary condition for velocity
    VectorVariable f;           // Forcing term (can vary in space)
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
    VectorVariable delta_u;                 // Velocity field
    VectorVariable xi;                  // x-direction solve intermediate
    VectorVariable eta;               // y-direction solve intermediate
    VectorVariable delta_eta;               // y-direction solve intermediate
    VectorVariable zeta;              // z-direction solve intermediate
    VectorVariable delta_zeta;              // z-direction solve intermediate
    VectorVariable gradient_pressure;   // ∇p correction term

    // ============================================================================
    // SCALAR LINEAR SOLVER VARIABLES (PRESSURE EQUATION AND OTHER SCALARS)
    // ============================================================================
    ScalarVariable pressure_predictor;  // Pressure predictor 
    ScalarVariable rhs;                 // RHS of scalar Poisson equation
    ScalarVariable psi;                 // Auxiliary scalar (potential or correction)
    ScalarVariable phi;                 // Pressure correction
    ScalarVariable other_phi;           // Additional scalar field for iterative updates

    ScalarVariable sol_linear_system;   // Temporary solution of scalar linear system
    ScalarVariable a;                   // Tridiagonal coefficient a (lower diag)
    ScalarVariable b;                   // Tridiagonal coefficient b (main diag)
    ScalarVariable c;                   // Tridiagonal coefficient c (upper diag)
    // ============================================================================
    // FINAL SOLUTION STORAGE
    // ============================================================================
    VectorVariable velocity_solution;  // Final converged velocity
    ScalarVariable pressure_solution; // Final converged pressure
};
