#include <string>
#include <cmath>
#include <functional>
#include <vector>
#include "ScalarVariable.hpp"
#include "VectorVariable.hpp"
#ifdef USE_MPI
#include "SolverCorrectMPI.hpp"
#include "MPICommunicator.hpp"
#else
#include "Solver.hpp"
#endif
#include "manufactured_solution_technique.hpp"

class NavierStokesBrinkman
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
    NavierStokesBrinkman(const Dim Nx, const Dim Ny, const Dim Nz,
                          const Real dt, const Real T,
                          BoundaryFunctions forcing_func,
                          std::function<Real(Real, Real, Real)> k_func,
                          std::string u_boundary_file,
                          std::string p_boundary_file,
                          BoundaryFunctions p_exact,
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
          forcing_function(forcing_func),
          k_function(k_func),
          nu(nu),
          k_field(Nx, Ny, Nz, dx, dy, dz),
          gamma_field(Nx, Ny, Nz, dx, dy, dz),

          p_boundary(),
          u_boundary(),
          p_exact(p_exact),

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
          pressure_solution(Nx, Ny, Nz, dx, dy, dz),
          use_openmp(use_openmp)

    {
        // --- Initialize all fields ---

        // // Initialize intermediate fields to zero (CRITICAL FIX)
        g.set_all(0.0f);

        vector_rhs.set_all(0.0f);
        vector_intermediate_solution.set_all(0.0f);
        xi.set_all(0.0f);
        eta.set_all(0.0f);
        zeta.set_all(0.0f);
        pressure_predictor.set_all(Real(0.0));
        rhs.set_all(0.0f);
        psi.set_all(0.0f);
        phi.set_all(0.0f);
        // other_phi.set_all(0.0f);

        // Final solution fields are set to initial conditions in solve()
        velocity_solution.set_all(0.0f);
        pressure_solution.set_all(0.0f);

        u_boundary.setParsing(u_boundary_file);
        p_boundary.setParsing(p_boundary_file);
        initialize_k_field();
        initialize_gamma_field();
        velocity_solver.u_boundary = u_boundary;
        pressure_solver.p_boundary = p_boundary;
        velocity_solver.gamma_field = gamma_field;
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
    void build_rhs(VectorVariable &rhs,
                   const VectorVariable &velocity_solution,
                   const ScalarVariable &p_star, // predictor pressure at cell centers
                   const VectorVariable &f_half,
                   Real dx, Real dy, Real dz,
                   Dim Nx, Dim Ny, Dim Nz,
                   Real dt,
                   Real nu,
                   const ScalarVariable &k);

    void center_pressure(ScalarVariable &pressure_field);
    void compute_divergence_cell_center(const VectorVariable &u,
                                        ScalarVariable &div);

    void compute_rhs_pressure(Real t);
    void update_pressure_and_velocity_fields();
#ifdef USE_MPI

    Real solve_mpi(const ManufacturedSolution &mms, const MPITopology3D &topo);
    void compute_forcing_term_mpi(VectorVariable &f, Real t, const MPITopology3D &topo);
    void globalize_velocity(VectorVariable &field, Dim Nx, Dim Ny, Dim Nz, const MPITopology3D &topo);
    void globalize_pressure(ScalarVariable &field, Dim Nx, Dim Ny, Dim Nz, const MPITopology3D &topo);

    void build_rhs_mpi(VectorVariable &rhs,
                       const VectorVariable &velocity_solution,
                       const ScalarVariable &p_star, // predictor pressure at cell centers
                       const VectorVariable &f_half,
                       Real dx, Real dy, Real dz,
                       Dim Nx, Dim Ny, Dim Nz,
                       Real dt,
                       Real nu,
                       const ScalarVariable &k, const MPITopology3D &topo);

    #ifdef USE_MPI
    /**
     * @brief Computes L2 errors for both Velocity and Pressure in a single pass over the grid.
     * * @param u_num Numerical velocity field (Pass nsb_solver.u_0)
     * @param p_num Numerical pressure field (Pass nsb_solver.p_0)
     * @param mms   Manufactured Solution instance
     * @param t     Current simulation time
     * @param topo  MPI Topology for bounds
     */
    std::pair<std::pair<Real, Real>, std::pair<Real, Real>> compute_L2_errors_mpi(
        const VectorVariable &u_num,
        const ScalarVariable &p_num,
        const ManufacturedSolution &mms,
        Real t,
        const MPITopology3D &topo)
    {
        // 1. Get Local Iteration Bounds (Global Indices)
        Dim i0 = topo.local_i0(Nx), i1 = topo.local_i1(Nx);
        Dim j0 = topo.local_j0(Ny), j1 = topo.local_j1(Ny);
        Dim k0 = topo.local_k0(Nz), k1 = topo.local_k1(Nz);

        Real local_sum_err_u = 0.0, local_sum_norm_u = 0.0;
        Real local_sum_err_p = 0.0, local_sum_norm_p = 0.0;

        // 2. Single Unified Loop for both Fields
        // Collapsing loops improves OpenMP efficiency
        #pragma omp parallel for reduction(+:local_sum_err_u, local_sum_norm_u, local_sum_err_p, local_sum_norm_p) collapse(2) if (use_openmp)
        for (Dim k = k0; k < k1; ++k)
        {
            for (Dim j = j0; j < j1; ++j)
            {
                for (Dim i = i0; i < i1; ++i)
                {
                    // --- Coordinates ---
                    Real x = i * dx;
                    Real y = j * dy;
                    Real z = k * dz;

                    // ============================
                    // 1. VELOCITY (Staggered Grid)
                    // ============================
                    // Fetch Exact Solution at staggered positions
                    Real u_ex_x = u_boundary.value<0>(x + dx * 0.5, y, z, t);
                    Real u_ex_y = u_boundary.value<1>(x, y + dy * 0.5, z, t);
                    Real u_ex_z = u_boundary.value<2>(x, y, z + dz * 0.5, t);

                    // Fetch Numerical Solution
                    // NOTE: Ensure value() accepts Global Indices (i,j,k)
                    Real u_num_x = u_num.value(0, i, j, k);
                    Real u_num_y = u_num.value(1, i, j, k);
                    Real u_num_z = u_num.value(2, i, j, k);

                    Real du_x = u_num_x - u_ex_x;
                    Real du_y = u_num_y - u_ex_y;
                    Real du_z = u_num_z - u_ex_z;

                    local_sum_err_u  += du_x*du_x + du_y*du_y + du_z*du_z;
                    local_sum_norm_u += u_ex_x*u_ex_x + u_ex_y*u_ex_y + u_ex_z*u_ex_z;

                    // ============================
                    // 2. PRESSURE (Cell Center)
                    // ============================
                    Real p_ex  = p_exact.value(x, y, z, t);
                    Real p_val = p_num.get(i, j, k);
                    Real dp    = p_val - p_ex;

                    local_sum_err_p  += dp * dp;
                    local_sum_norm_p += p_ex * p_ex;
                }
            }
        }

        // 3. Global Reduction (Pack 4 values into one MPI call)
        // Order: [err_u_sq, norm_u_sq, err_p_sq, norm_p_sq]
        Real local_sums[4] = {local_sum_err_u, local_sum_norm_u, local_sum_err_p, local_sum_norm_p};
        Real global_sums[4] = {0.0};

        // Determine MPI type (float or double)
        MPI_Datatype mpi_real = (sizeof(Real) == sizeof(double)) ? MPI_DOUBLE : MPI_FLOAT;
        
        MPI_Allreduce(local_sums, global_sums, 4, mpi_real, MPI_SUM, topo.cart_comm());

        // 4. Final Calculation
        Real dV = dx * dy * dz;
        
        Real err_u_abs = std::sqrt(global_sums[0] * dV);
        Real nrm_u_abs = std::sqrt(global_sums[1] * dV);
        
        Real err_p_abs = std::sqrt(global_sums[2] * dV);
        Real nrm_p_abs = std::sqrt(global_sums[3] * dV);

        // Prevent division by zero
        Real err_u_rel = (nrm_u_abs > 1e-12) ? err_u_abs / nrm_u_abs : err_u_abs;
        Real err_p_rel = (nrm_p_abs > 1e-12) ? err_p_abs / nrm_p_abs : err_p_abs;

        return {{err_u_abs, err_u_rel}, {err_p_abs, err_p_rel}};
    }
#endif

#else
    Real solve(const ManufacturedSolution &mms);
#endif

    /**
     * @brief Computes the L2 relative error between the numerical solution and the MMS exact solution.
     * * @param u_num Numerical velocity field (from Solver)
     * @param p_num Numerical pressure field (from Solver)
     * @param mms   Manufactured Solution instance (provides exact solution and grid spacing)
     * @param t     Time at which to evaluate the error
     * @return std::pair<Real, Real> {Relative Error Velocity, Relative Error Pressure}
     */
    std::pair<std::pair<Real, Real>, std::pair<Real, Real>> compute_L2_errors(
        const VectorVariable &u_num,
        const ScalarVariable &p_num,
        const ManufacturedSolution &mms,
        Real t)
    {

        Real err_u = 0.0, err_p = 0.0;
        Real norm_u = 0.0, norm_p = 0.0;
        Real dV = dx * dy * dz;

        for (Dim k = 0; k < Nz; ++k)
        {
            for (Dim j = 0; j < Ny; ++j)
            {
                for (Dim i = 0; i < Nx; ++i)
                {
                    Real x = i * dx;
                    Real y = j * dy;
                    Real z = k * dz;

                    auto u_ex_x = u_boundary.value<0>(x + dx * 0.5, y, z, t);
                    auto u_ex_y = u_boundary.value<1>(x, y + dy * 0.5, z, t);
                    auto u_ex_z = u_boundary.value<2>(x, y, z + dz * 0.5, t);
                    std::vector<Real> u_ex = {u_ex_x, u_ex_y, u_ex_z};

                    Real p_ex = p_exact.value(x, y, z, t);
                    Real ux = velocity_solution.value(0, i, j, k);
                    Real uy = velocity_solution.value(1, i, j, k);
                    Real uz = velocity_solution.value(2, i, j, k);
                    Real pN = pressure_solution.get(i, j, k);

                    Real dux = ux - u_ex[0];
                    Real duy = uy - u_ex[1];
                    Real duz = uz - u_ex[2];

                    err_u += dux * dux + duy * duy + duz * duz;
                    norm_u += u_ex[0] * u_ex[0] + u_ex[1] * u_ex[1] + u_ex[2] * u_ex[2];

                    err_p += (pN - p_ex) * (pN - p_ex);
                    norm_p += p_ex * p_ex;
                }
            }
        }

        err_u = std::sqrt(err_u * dV);
        norm_u = std::sqrt(norm_u * dV);
        err_p = std::sqrt(err_p * dV);
        norm_p = std::sqrt(norm_p * dV);

        Real rel_err_u = (norm_u > 1e-12) ? err_u / norm_u : err_u;
        Real rel_err_p = (norm_p > 1e-12) ? err_p / norm_p : err_p;

        // 6. PRINT DETAILED DIAGNOSTICS (Crucial for debugging)
        // std::cout << "------------------------------------------\n";
        // std::cout << " ERRORS at t = " << t << "\n";
        // std::cout << " Velocity -> Abs: " << err_u << " | Ref Norm: " << norm_u << " | Rel: " << rel_err_u << "\n";
        // std::cout << " Pressure -> Abs: " << err_p << " | Ref Norm: " << norm_p << " | Rel: " << rel_err_p << "\n";
        // std::cout << "------------------------------------------\n";

        return {{err_u, rel_err_u}, {err_p, rel_err_p}};
    }

    void write_velocity_vtk(const std::string &filename) const;
    void write_pressure_vtk(const std::string &filename) const;

    // ============================================================================
    // GRID, MATERIAL, AND TIME INFORMATION
    // ============================================================================
    Grid grid; // Grid geometry and domain decomposition

    Real dt; // Time step

    Dim Nx; // Grid points in x
    Dim Ny; // Grid points in y
    Dim Nz; // Grid points in z

    Real dx = Real(1.0); // Grid spacing in x
    Real dy = Real(1.0); // Grid spacing in y
    Real dz = Real(1.0); // Grid spacing in z

    Real T;

    // ============================================================================
    // SOLVER CLASS
    // ============================================================================
    VelocitySolver velocity_solver;
    PressureSolver pressure_solver;

    // ============================================================================
    // PHYSICAL AND MATERIAL FIELDS
    // ============================================================================
    VectorVariable u_0;                               // Velocity field
    ScalarVariable p_0;                               // Pressure field
    BoundaryFunctions forcing_function;               // Forcing term (can vary in space)
    std::function<Real(Real, Real, Real)> k_function; // Forcing term (can vary in space)
    Real nu;                                          // Kinematic viscosity (can vary in space)
    ScalarVariable k_field;                           // Brinkman permeability or resistance term
    ScalarVariable gamma_field;                       // Gamma field for Brinkman term
    BoundaryFunctions p_boundary;                     // Boundary condition for pressure
    BoundaryFunctions u_boundary;                     // Boundary condition for velocity
    BoundaryFunctions p_exact;                        // Exact pressure for MMS

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
