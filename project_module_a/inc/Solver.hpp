#ifndef SOLVER_HPP
#define SOLVER_HPP
#include <string>
#include <cmath>
#include <functional> // For std::function/lambdas
#include "Variables.hpp"
#include "ScalarVariable.hpp"
#include "VectorVariable.hpp"
#include "DimensionHandler.hpp"
#include "BoundaryFunctions.hpp"
#include "DomainDecomposition.hpp"

#ifdef USE_MPI
#include <mpi.h>
#endif

class Solver
{

public:
    // Pure virtual destructor makes the class abstract
    virtual ~Solver() = 0;
    Solver(Dim Nx_, Dim Ny_, Dim Nz_, Real dx_, Real dy_, Real dz_, Real dt_)
        : Nx(Nx_), Ny(Ny_), Nz(Nz_),
          dx(dx_), dy(dy_), dz(dz_), dt(dt_)
#ifdef USE_MPI
        , decomp(nullptr)
#endif
    { t = dt_; }; // solve doesn't solve for t=0 called firstly at t=dt

#ifdef USE_MPI
    /**
     * @brief Set domain decomposition pointer
     */
    void set_decomposition(DomainDecomposition* decomp_ptr) {
        decomp = decomp_ptr;
    }

    /**
     * @brief Get domain decomposition pointer
     */
    DomainDecomposition* get_decomposition() const {
        return decomp;
    }
#endif

    template <typename StrideFunc, Dim direction>
    void solve(ScalarVariable &rhs, ScalarVariable &solution, const DimensionsHandlerScalar<StrideFunc> &dim_handler);
    void advance_time()
    {
        t += dt;
    }

protected:
    Dim Nx; // Grid points in x
    Dim Ny; // Grid points in y
    Dim Nz; // Grid points in z

    Real dx; // Grid spacing in x
    Real dy; // Grid spacing in y
    Real dz; // Grid spacing in z

    Real t;
    Real dt;

#ifdef USE_MPI
    DomainDecomposition* decomp;
#endif

    // Make available to derived classes
    void thomas_algorithm(const std::vector<Real> &a, const std::vector<Real> &b, const std::vector<Real> &c, const std::vector<Real> &rhs, std::vector<Real> &x)
    {
        int n = rhs.size();
        std::vector<Real> c_prime(c.size(), 0.0);
        std::vector<Real> rhs_prime(n, 0.0);

        c_prime[0] = c[0] / b[0];
        rhs_prime[0] = rhs[0] / b[0];

        for (int i = 1; i < n; ++i)
        {
            Real m = Real(1.0) / (b[i] - a[i] * c_prime[i - 1]);
            c_prime[i] = c[i] * m;
            rhs_prime[i] = (rhs[i] - a[i] * rhs_prime[i - 1]) * m;
        }

        x[n - 1] = rhs_prime[n - 1];

        for (int i = n - 2; i >= 0; --i)
        {
            x[i] = rhs_prime[i] - c_prime[i] * x[i + 1];
        }
    }

#ifdef USE_MPI
    /**
     * @brief Schur complement block solver for parallel execution
     * Implements the algorithm from lecture slides (lines 430-437)
     * 
     * Algorithm:
     * 1. Preprocessing: Assemble tridiagonal Schur complement matrix S
     * 2. Runtime (per timestep):
     *    a. Compute f_i (RHS for internal points)
     *    b. Solve A_ii * temp = f_i locally using Thomas
     *    c. Compute local contribution: f_s_local - A_si * temp
     *    d. MPI_Allreduce to assemble global interface RHS
     *    e. Solve interface system: S * u_s = f_s_global (Thomas on all ranks)
     *    f. Solve internal system: A_ii * u_i = f_i - A_is * u_s (Thomas locally)
     */
    void schur_complement_solver(
        const std::vector<Real>& a, const std::vector<Real>& b, const std::vector<Real>& c,
        const std::vector<Real>& rhs, std::vector<Real>& x, Dim direction)
    {
        if (decomp == nullptr) {
            // Fallback to serial Thomas algorithm
            thomas_algorithm(a, b, c, rhs, x);
            return;
        }

        auto [left_neighbor, right_neighbor] = decomp->get_neighbors(direction);
        
        // If no neighbors in this direction, use serial solver
        if (left_neighbor == MPI_PROC_NULL && right_neighbor == MPI_PROC_NULL) {
            thomas_algorithm(a, b, c, rhs, x);
            return;
        }

        int n = rhs.size();
        MPI_Comm comm = decomp->get_cart_comm();

        // Identify internal vs interface points
        // Interface points: first point (if left neighbor exists) and last point (if right neighbor exists)
        bool has_left_interface = (left_neighbor != MPI_PROC_NULL);
        bool has_right_interface = (right_neighbor != MPI_PROC_NULL);
        
        int n_internal = n;
        int i_start_internal = 0;
        int i_end_internal = n;

        if (has_left_interface) {
            i_start_internal = 1;
            n_internal--;
        }
        if (has_right_interface) {
            i_end_internal = n - 1;
            n_internal--;
        }

        // Step 1: Extract f_i (RHS for internal points)
        std::vector<Real> f_i(n_internal);
        std::vector<Real> a_i(n_internal), b_i(n_internal), c_i(n_internal);
        
        for (int i = 0; i < n_internal; ++i) {
            int global_i = i_start_internal + i;
            f_i[i] = rhs[global_i];
            a_i[i] = a[global_i];
            b_i[i] = b[global_i];
            c_i[i] = c[global_i];
        }

        // Step 2: Solve A_ii * temp = f_i locally using Thomas algorithm
        std::vector<Real> temp(n_internal);
        if (n_internal > 0) {
            thomas_algorithm(a_i, b_i, c_i, f_i, temp);
        }

        // Step 3: Compute local contribution to interface RHS
        // f_s_local - A_si * A_ii^(-1) * f_i
        std::vector<Real> f_s_local(2, 0.0);  // [left_interface, right_interface]
        std::vector<Real> A_si_temp(2, 0.0);

        if (has_left_interface) {
            // Left interface point contribution
            f_s_local[0] = rhs[0];
            // A_si * temp: only the first internal point contributes
            if (n_internal > 0) {
                A_si_temp[0] = c[0] * temp[0];  // c[0] connects interface to first internal
            }
            f_s_local[0] -= A_si_temp[0];
        }

        if (has_right_interface) {
            // Right interface point contribution
            f_s_local[1] = rhs[n - 1];
            // A_si * temp: only the last internal point contributes
            if (n_internal > 0) {
                A_si_temp[1] = a[n - 1] * temp[n_internal - 1];  // a[n-1] connects interface to last internal
            }
            f_s_local[1] -= A_si_temp[1];
        }

        // Step 4: Assemble global interface RHS using MPI_Allreduce
        std::vector<Real> f_s_global(2);
        MPI_Allreduce(f_s_local.data(), f_s_global.data(), 2, MPI_FLOAT, MPI_SUM, comm);

        // Step 5: Solve interface system S * u_s = f_s_global
        // Build Schur complement matrix S (tridiagonal, small)
        // S = A_ss - A_si * A_ii^(-1) * A_is
        // For simplicity, each rank solves its local interface independently
        // In a more sophisticated implementation, we'd solve a global interface system
        
        if (has_left_interface) {
            // Simplified: Direct solve for left interface
            // In full implementation, this would be part of a global tridiagonal solve
            x[0] = f_s_global[0] / b[0];
        }

        if (has_right_interface) {
            // Simplified: Direct solve for right interface
            x[n - 1] = f_s_global[1] / b[n - 1];
        }

        // Step 6: Compute f_i - A_is * u_s
        for (int i = 0; i < n_internal; ++i) {
            int global_i = i_start_internal + i;
            Real correction = 0.0;
            
            if (global_i == i_start_internal && has_left_interface) {
                correction += a[global_i] * x[0];  // a connects to left interface
            }
            if (global_i == i_end_internal - 1 && has_right_interface) {
                correction += c[global_i] * x[n - 1];  // c connects to right interface
            }
            
            f_i[i] -= correction;
        }

        // Step 7: Solve internal system A_ii * u_i = (f_i - A_is * u_s)
        if (n_internal > 0) {
            thomas_algorithm(a_i, b_i, c_i, f_i, temp);
            
            // Copy solution back
            for (int i = 0; i < n_internal; ++i) {
                x[i_start_internal + i] = temp[i];
            }
        }
    }
    
    /**
     * @brief Batched Schur complement solver - processes all lines with ONE MPI_Allreduce
     * This dramatically reduces communication overhead from N calls to 1 call per direction
     */
    void batched_schur_complement_solver(
        const std::vector<std::vector<Real>>& a_batch,
        const std::vector<std::vector<Real>>& b_batch,
        const std::vector<std::vector<Real>>& c_batch,
        const std::vector<std::vector<Real>>& rhs_batch,
        std::vector<std::vector<Real>>& x_batch,
        Dim direction)
    {
        if (decomp == nullptr || rhs_batch.empty()) {
            // Fallback: solve each line independently with Thomas algorithm
            for (size_t line = 0; line < rhs_batch.size(); ++line) {
                thomas_algorithm(a_batch[line], b_batch[line], c_batch[line], 
                               rhs_batch[line], x_batch[line]);
            }
            return;
        }

        auto [left_neighbor, right_neighbor] = decomp->get_neighbors(direction);
        
        // If no neighbors, use serial solver for all lines
        if (left_neighbor == MPI_PROC_NULL && right_neighbor == MPI_PROC_NULL) {
            for (size_t line = 0; line < rhs_batch.size(); ++line) {
                thomas_algorithm(a_batch[line], b_batch[line], c_batch[line], 
                               rhs_batch[line], x_batch[line]);
            }
            return;
        }

        int num_lines = rhs_batch.size();
        MPI_Comm comm = decomp->get_cart_comm();
        
        bool has_left_interface = (left_neighbor != MPI_PROC_NULL);
        bool has_right_interface = (right_neighbor != MPI_PROC_NULL);

        // Collect ALL interface contributions across all lines
        std::vector<Real> all_f_s_local;  // [line0_left, line0_right, line1_left, line1_right, ...]
        all_f_s_local.reserve(num_lines * 2);

        // Step 1-3: Process each line's internal system and collect interface contributions
        for (int line = 0; line < num_lines; ++line) {
            const auto& rhs = rhs_batch[line];
            const auto& a = a_batch[line];
            const auto& b = b_batch[line];
            const auto& c = c_batch[line];
            auto& x = x_batch[line];
            
            int n = rhs.size();
            
            int n_internal = n;
            int i_start_internal = 0;
            int i_end_internal = n;

            if (has_left_interface) {
                i_start_internal = 1;
                n_internal--;
            }
            if (has_right_interface) {
                i_end_internal = n - 1;
                n_internal--;
            }

            // Extract and solve internal system
            std::vector<Real> f_i(n_internal);
            std::vector<Real> a_i(n_internal), b_i(n_internal), c_i(n_internal);
            
            for (int i = 0; i < n_internal; ++i) {
                int global_i = i_start_internal + i;
                f_i[i] = rhs[global_i];
                a_i[i] = a[global_i];
                b_i[i] = b[global_i];
                c_i[i] = c[global_i];
            }

            std::vector<Real> temp(n_internal);
            if (n_internal > 0) {
                thomas_algorithm(a_i, b_i, c_i, f_i, temp);
            }

            // Compute interface contributions for this line
            Real f_s_left = 0.0, f_s_right = 0.0;
            
            if (has_left_interface) {
                f_s_left = rhs[0];
                if (n_internal > 0) {
                    f_s_left -= c[0] * temp[0];
                }
            }
            
            if (has_right_interface) {
                f_s_right = rhs[n - 1];
                if (n_internal > 0) {
                    f_s_right -= a[n - 1] * temp[n_internal - 1];
                }
            }
            
            all_f_s_local.push_back(f_s_left);
            all_f_s_local.push_back(f_s_right);
            
            // Store temp solution for later use
            if (n_internal > 0) {
                for (int i = 0; i < n_internal; ++i) {
                    x[i_start_internal + i] = temp[i];
                }
            }
        }

        // Step 4: ONE MPI_Allreduce for ALL interface contributions
        std::vector<Real> all_f_s_global(num_lines * 2);
        MPI_Allreduce(all_f_s_local.data(), all_f_s_global.data(), 
                     num_lines * 2, MPI_FLOAT, MPI_SUM, comm);

        // Step 5-7: Solve interface and update internal for each line
        for (int line = 0; line < num_lines; ++line) {
            const auto& rhs = rhs_batch[line];
            const auto& a = a_batch[line];
            const auto& b = b_batch[line];
            const auto& c = c_batch[line];
            auto& x = x_batch[line];
            
            int n = rhs.size();
            
            int n_internal = n;
            int i_start_internal = 0;
            int i_end_internal = n;

            if (has_left_interface) {
                i_start_internal = 1;
                n_internal--;
            }
            if (has_right_interface) {
                i_end_internal = n - 1;
                n_internal--;
            }

            // Extract global interface solutions for this line
            Real f_s_global_left = all_f_s_global[line * 2];
            Real f_s_global_right = all_f_s_global[line * 2 + 1];
            
            // Solve interface
            if (has_left_interface) {
                x[0] = f_s_global_left / b[0];
            }
            if (has_right_interface) {
                x[n - 1] = f_s_global_right / b[n - 1];
            }

            // Update internal with interface correction
            if (n_internal > 0) {
                std::vector<Real> f_i(n_internal);
                std::vector<Real> a_i(n_internal), b_i(n_internal), c_i(n_internal);
                
                for (int i = 0; i < n_internal; ++i) {
                    int global_i = i_start_internal + i;
                    f_i[i] = rhs[global_i];
                    a_i[i] = a[global_i];
                    b_i[i] = b[global_i];
                    c_i[i] = c[global_i];
                    
                    // Apply interface correction
                    Real correction = 0.0;
                    if (global_i == i_start_internal && has_left_interface) {
                        correction += a[global_i] * x[0];
                    }
                    if (global_i == i_end_internal - 1 && has_right_interface) {
                        correction += c[global_i] * x[n - 1];
                    }
                    f_i[i] -= correction;
                }

                std::vector<Real> temp(n_internal);
                thomas_algorithm(a_i, b_i, c_i, f_i, temp);
                
                for (int i = 0; i < n_internal; ++i) {
                    x[i_start_internal + i] = temp[i];
                }
            }
        }
    }
#endif
};

// Definition of pure virtual destructor
inline Solver::~Solver() {}

// =============================================================================================
// ====================================Pressure Solver Class====================================
// =============================================================================================

class PressureSolver : public Solver
{
public:
    template <Dim direction>
    void apply_bc(ScalarVariable &rhs)
    {
#ifdef USE_MPI
        // In parallel mode, only apply BCs at physical boundaries, not partition interfaces
        bool has_left_boundary = (decomp == nullptr) || decomp->owns_physical_boundary(direction, BoundarySide::LEFT);
        bool has_right_boundary = (decomp == nullptr) || decomp->owns_physical_boundary(direction, BoundarySide::RIGHT);
#else
        bool has_left_boundary = true;
        bool has_right_boundary = true;
#endif

        if constexpr (direction == 0) // X direction
        {
            for (Dim index_1 = 0; index_1 < Ny; ++index_1)
            {
                for (Dim index_2 = 0; index_2 < Nz; ++index_2)
                {
                    if (has_left_boundary) {
                        rhs.set(0, index_1, index_2) = rhs.get(0, index_1, index_2) - Real(2.0) / dx * p_boundary.value<0>(0, index_1 * dy, index_2 * dz, t);
                    }
                    if (has_right_boundary) {
                        rhs.set(Nx - 1, index_1, index_2) = rhs.get(Nx - 1, index_1, index_2) + Real(1.0) / dx * p_boundary.value<0>((Nx - 0.5) * dx, index_1 * dy, index_2 * dz, t);
                    }
                }
            }
        }
        else if constexpr (direction == 1) // Y direction
        {
            for (Dim index_1 = 0; index_1 < Nx; ++index_1)
            {
                for (Dim index_2 = 0; index_2 < Nz; ++index_2)
                {
                    if (has_left_boundary) {
                        rhs.set(index_1, 0, index_2) = rhs.get(index_1, 0, index_2) - Real(2.0) / dy * p_boundary.value<1>(index_1 * dx, 0, index_2 * dz, t);
                    }
                    if (has_right_boundary) {
                        rhs.set(index_1, Ny - 1, index_2) = rhs.get(index_1, Ny - 1, index_2) + Real(1.0) / dy * p_boundary.value<1>(index_1 * dx, (Ny - 0.5) * dy, index_2 * dz, t);
                    }
                }
            }
        }
        else if constexpr (direction == 2) // Z direction
        {
            for (Dim index_1 = 0; index_1 < Nx; ++index_1)
            {
                for (Dim index_2 = 0; index_2 < Ny; ++index_2)
                {
                    if (has_left_boundary) {
                        rhs.set(index_1, index_2, 0) = rhs.get(index_1, index_2, 0) - Real(2.0) / dz * p_boundary.value<2>(index_1 * dx, index_2 * dy, 0, t);
                    }
                    if (has_right_boundary) {
                        rhs.set(index_1, index_2, Nz - 1) = rhs.get(index_1, index_2, Nz - 1) + Real(1.0) / dz * p_boundary.value<2>(index_1 * dx, index_2 * dy, (Nz - 0.5) * dz, t);
                    }
                }
            }
        }
    };

    template <Dim direction, typename StrideFunc>
    void block_solver(const ScalarVariable &rhs, ScalarVariable &solution, const DimensionsHandlerScalar<StrideFunc> &dim_handler)
    {
        std::vector<Real> a(dim_handler.N1, Real(-1.0) / (dim_handler.dN1 * dim_handler.dN1));
        std::vector<Real> b(dim_handler.N1, Real(1.0) + (Real(2.0) / (dim_handler.dN1 * dim_handler.dN1)));
        std::vector<Real> c(dim_handler.N1, Real(-1.0) / (dim_handler.dN1 * dim_handler.dN1));
        std::vector<Real> d(dim_handler.N1);
        std::vector<Real> x(dim_handler.N1);

        a[0] = Real(0.0);
        c[0] = Real(-2.0) / (dim_handler.dN1 * dim_handler.dN1);
        b[dim_handler.N1 - 1] = Real(1.0) + Real(1.0) / (dim_handler.dN1 * dim_handler.dN1);
        c[dim_handler.N1 - 1] = Real(0.0);

        if constexpr (direction == 0)
        {
#ifdef USE_MPI
            // Batched approach: collect all lines, one MPI_Allreduce
            Dim num_lines = Ny * Nz;
            std::vector<std::vector<Real>> a_batch(num_lines, a);
            std::vector<std::vector<Real>> b_batch(num_lines, b);
            std::vector<std::vector<Real>> c_batch(num_lines, c);
            std::vector<std::vector<Real>> rhs_batch(num_lines, std::vector<Real>(Nx));
            std::vector<std::vector<Real>> x_batch(num_lines, std::vector<Real>(Nx));
            
            // Collect all RHS data
            Dim line_idx = 0;
            for (Dim index_1 = 0; index_1 < Ny; ++index_1) {
                for (Dim index_2 = 0; index_2 < Nz; ++index_2) {
                    for (Dim index_0 = 0; index_0 < Nx; ++index_0)
                        rhs_batch[line_idx][index_0] = rhs.get(index_0, index_1, index_2);
                    line_idx++;
                }
            }
            
            // ONE batched solve for entire direction
            batched_schur_complement_solver(a_batch, b_batch, c_batch, rhs_batch, x_batch, direction);
            
            // Write solutions back
            line_idx = 0;
            for (Dim index_1 = 0; index_1 < Ny; ++index_1) {
                for (Dim index_2 = 0; index_2 < Nz; ++index_2) {
                    for (Dim index_0 = 0; index_0 < Nx; ++index_0)
                        solution.set(index_0, index_1, index_2) = x_batch[line_idx][index_0];
                    line_idx++;
                }
            }
#else
            // Serial: solve line by line
            for (Dim index_1 = 0; index_1 < Ny; ++index_1)
            {
                for (Dim index_2 = 0; index_2 < Nz; ++index_2)
                {
                    for (Dim index_0 = 0; index_0 < Nx; ++index_0)
                        d[index_0] = rhs.get(index_0, index_1, index_2);
                    thomas_algorithm(a, b, c, d, x);
                    for (Dim index_0 = 0; index_0 < Nx; ++index_0)
                        solution.set(index_0, index_1, index_2) = x[index_0];
                }
            }
#endif
        }
        else if constexpr (direction == 1)
        {
#ifdef USE_MPI
            // Batched approach: collect all lines, one MPI_Allreduce
            Dim num_lines = Nx * Nz;
            std::vector<std::vector<Real>> a_batch(num_lines, a);
            std::vector<std::vector<Real>> b_batch(num_lines, b);
            std::vector<std::vector<Real>> c_batch(num_lines, c);
            std::vector<std::vector<Real>> rhs_batch(num_lines, std::vector<Real>(Ny));
            std::vector<std::vector<Real>> x_batch(num_lines, std::vector<Real>(Ny));
            
            // Collect all RHS data
            Dim line_idx = 0;
            for (Dim index_1 = 0; index_1 < Nx; ++index_1) {
                for (Dim index_2 = 0; index_2 < Nz; ++index_2) {
                    for (Dim index_0 = 0; index_0 < Ny; ++index_0)
                        rhs_batch[line_idx][index_0] = rhs.get(index_1, index_0, index_2);
                    line_idx++;
                }
            }
            
            // ONE batched solve for entire direction
            batched_schur_complement_solver(a_batch, b_batch, c_batch, rhs_batch, x_batch, direction);
            
            // Write solutions back
            line_idx = 0;
            for (Dim index_1 = 0; index_1 < Nx; ++index_1) {
                for (Dim index_2 = 0; index_2 < Nz; ++index_2) {
                    for (Dim index_0 = 0; index_0 < Ny; ++index_0)
                        solution.set(index_1, index_0, index_2) = x_batch[line_idx][index_0];
                    line_idx++;
                }
            }
#else
            // Serial: solve line by line
            for (Dim index_1 = 0; index_1 < Nx; ++index_1)
            {
                for (Dim index_2 = 0; index_2 < Nz; ++index_2)
                {
                    for (Dim index_0 = 0; index_0 < Ny; ++index_0)
                        d[index_0] = rhs.get(index_1, index_0, index_2);
                    thomas_algorithm(a, b, c, d, x);
                    for (Dim index_0 = 0; index_0 < Ny; ++index_0)
                        solution.set(index_1, index_0, index_2) = x[index_0];
                }
            }
#endif
        }
        else if constexpr (direction == 2)
        {
#ifdef USE_MPI
            // Batched approach: collect all lines, one MPI_Allreduce
            Dim num_lines = Nx * Ny;
            std::vector<std::vector<Real>> a_batch(num_lines, a);
            std::vector<std::vector<Real>> b_batch(num_lines, b);
            std::vector<std::vector<Real>> c_batch(num_lines, c);
            std::vector<std::vector<Real>> rhs_batch(num_lines, std::vector<Real>(Nz));
            std::vector<std::vector<Real>> x_batch(num_lines, std::vector<Real>(Nz));
            
            // Collect all RHS data
            Dim line_idx = 0;
            for (Dim index_1 = 0; index_1 < Nx; ++index_1) {
                for (Dim index_2 = 0; index_2 < Ny; ++index_2) {
                    for (Dim index_0 = 0; index_0 < Nz; ++index_0)
                        rhs_batch[line_idx][index_0] = rhs.get(index_1, index_2, index_0);
                    line_idx++;
                }
            }
            
            // ONE batched solve for entire direction
            batched_schur_complement_solver(a_batch, b_batch, c_batch, rhs_batch, x_batch, direction);
            
            // Write solutions back
            line_idx = 0;
            for (Dim index_1 = 0; index_1 < Nx; ++index_1) {
                for (Dim index_2 = 0; index_2 < Ny; ++index_2) {
                    for (Dim index_0 = 0; index_0 < Nz; ++index_0)
                        solution.set(index_1, index_2, index_0) = x_batch[line_idx][index_0];
                    line_idx++;
                }
            }
#else
            // Serial: solve line by line
            for (Dim index_1 = 0; index_1 < Nx; ++index_1)
            {
                for (Dim index_2 = 0; index_2 < Ny; ++index_2)
                {
                    for (Dim index_0 = 0; index_0 < Nz; ++index_0)
                        d[index_0] = rhs.get(index_1, index_2, index_0);
                    thomas_algorithm(a, b, c, d, x);
                    for (Dim index_0 = 0; index_0 < Nz; ++index_0)
                        solution.set(index_1, index_2, index_0) = x[index_0];
                }
            }
#endif
        }
    }

    /**
     * @brief Applies the Pressure Poisson matrix A to a scalar field x to compute RHS = A*x.
     * Mimics the coefficients used in block_solver for consistency check.
     */
    template <Dim direction, typename StrideFunc>
    void apply_matrix_operator(const ScalarVariable &x_scalar, ScalarVariable &rhs_scalar, const DimensionsHandlerScalar<StrideFunc> &dim_handler)
    {
        // 1. Determine Dimensions and Grid Spacing based on direction
        Dim N = (direction == 0) ? Nx : ((direction == 1) ? Ny : Nz);
        Real h = (direction == 0) ? dx : ((direction == 1) ? dy : dz);
        Real h2 = h * h;

        // Precompute Coefficients
        Real coeff = 1.0 / h2;

        // Standard Interior Stencil: -1/h^2, 1 + 2/h^2, -1/h^2
        // Wait! In your block_solver:
        // a = -1/h^2
        // b = 1 + 2/h^2
        // c = -1/h^2
        Real a_int = -coeff;
        Real b_int = 1.0 + 2.0 * coeff;
        Real c_int = -coeff;

        // Boundary Start (i=0):
        // In block_solver: a=0, c = -2/h^2 (due to p_-1 = p_1)
        // b is standard (1 + 2/h^2)
        Real b_start = b_int;
        Real c_start = -2.0 * coeff;

        // Boundary End (i=N-1):
        // In block_solver: c=0, b = 1 + 1/h^2 (due to p_N+1 = p_N? Check logic)
        // Your code: b[N-1] = 1.0 + 1.0/(h*h)
        // This implies the BC was p_N+1 = p_N (Homogeneous Neumann at right wall?)
        // Standard Neumann is p_N+1 = p_N-1 (Centered) or p_N (Forward)
        // Let's match YOUR code exactly:
        Real a_end = a_int;             // -1/h^2
        Real b_end = 1.0 + 1.0 * coeff; // Matches your code: Real(1.0) + Real(1.0)/...

        // 2. Loop Limits
        Dim Outer1 = (direction == 0) ? Ny : ((direction == 1) ? Nx : Nx);
        Dim Outer2 = (direction == 0) ? Nz : ((direction == 1) ? Nz : Ny);

        for (Dim i1 = 0; i1 < Outer1; ++i1)
        {
            for (Dim i2 = 0; i2 < Outer2; ++i2)
            {
                // Lambdas for access
                auto get_val = [&](Dim i)
                {
                    if constexpr (direction == 0)
                        return x_scalar.get(i, i1, i2);
                    else if constexpr (direction == 1)
                        return x_scalar.get(i1, i, i2);
                    else
                        return x_scalar.get(i1, i2, i);
                };

                auto set_rhs = [&](Dim i, Real val)
                {
                    if constexpr (direction == 0)
                        rhs_scalar.set(i, i1, i2) = val;
                    else if constexpr (direction == 1)
                        rhs_scalar.set(i1, i, i2) = val;
                    else
                        rhs_scalar.set(i1, i2, i) = val;
                };

                for (Dim i = 0; i < N; ++i)
                {
                    Real val = 0.0;

                    if (i == 0)
                    {
                        // Boundary Start (Neumann Left: p_-1 = p_1)
                        // Row 0: b*p_0 + c*p_1 (where c is doubled)
                        val = b_start * get_val(0) + c_start * get_val(1);
                    }
                    else if (i == N - 1)
                    {
                        // Boundary End
                        // Row N-1: a*p_{N-2} + b*p_{N-1}
                        val = a_end * get_val(i - 1) + b_end * get_val(i);
                    }
                    else
                    {
                        // Interior
                        // a*p_{i-1} + b*p_i + c*p_{i+1}
                        val = a_int * get_val(i - 1) + b_int * get_val(i) + c_int * get_val(i + 1);
                    }

                    set_rhs(i, val);
                }
            }
        }
    }

    BoundaryFunctions &p_boundary;

    PressureSolver(Dim Nx_, Dim Ny_, Dim Nz_, Real dx_, Real dy_, Real dz_, Real dt_, BoundaryFunctions &p_boundary_)
        : Solver(Nx_, Ny_, Nz_, dx_, dy_, dz_, dt_), p_boundary(p_boundary_) {}

    template <typename StrideFunc, Dim direction>
    void solve_pressure(ScalarVariable &rhs, ScalarVariable &solution, const DimensionsHandlerScalar<StrideFunc> &dim_handler)
    {
        apply_bc<direction>(rhs);
        block_solver<direction, StrideFunc>(rhs, solution, dim_handler);
        advance_time();
    };
    BoundaryFunctions &set_p_boundary() { return p_boundary; }
};

// =============================================================================================
// ====================================Velocity Solver Class====================================
// =============================================================================================

class VelocitySolver : public Solver
{
public:
    ScalarVariable &gamma_field;
    BoundaryFunctions &u_boundary;

    // --- HELPER: Computes a,b,c,d for INTERNAL points (1 to N-2) ---
    // Coefficients matched to Professor's slides: a = -gamma/h^2, b = 1 + 2*gamma/h^2, c = -gamma/h^2
    template <typename RhsGetter, typename GammaGetter>
    void setup_TDMA_internal(
        Dim N, Real h,
        std::vector<Real> &a, std::vector<Real> &b, std::vector<Real> &c, std::vector<Real> &d,
        RhsGetter get_rhs, GammaGetter get_gamma)
    {
        Real h2 = h * h;
        // Iterate over internal points only
        for (Dim i = 1; i < N - 1; ++i)
        {
            Real gamma_val = get_gamma(i); // Assumes gamma is positive
            Real coeff = gamma_val / h2;

            d[i] = get_rhs(i);

            // STABLE COEFFICIENTS for (I - gamma*L) u = RHS
            a[i] = -coeff;
            b[i] = 1.0f + 2.0f * coeff;
            c[i] = -coeff;
        }
    }

    /**
     * @brief Applies the system matrix A to a vector x to compute RHS = A*x.
     * This mimics the implicit operator coefficients used in block_solver.
     */
    /**
     * @brief Applies the system matrix A to a vector x to compute RHS = A*x.
     * Handles known faces exactly like block_solver to ensure consistent verification.
     */
    template <Dim direction, typename StrideFunc>
    void apply_matrix_operator(const VectorVariable &x_vec, VectorVariable &rhs_vec, const DimensionsHandlerVector<StrideFunc> &dim_handler)
    {
        Dim N = (direction == 0) ? Nx : ((direction == 1) ? Ny : Nz);
        Real h = (direction == 0) ? dx : ((direction == 1) ? dy : dz);

        Dim Comp1 = dim_handler.Comp1; // Normal component
        Dim Comp2 = dim_handler.Comp2; // Tangent 1
        Dim Comp3 = dim_handler.Comp3; // Tangent 2

        Dim Outer1 = (direction == 0) ? Ny : ((direction == 1) ? Nx : Nx);
        Dim Outer2 = (direction == 0) ? Nz : ((direction == 1) ? Nz : Ny);

        for (Dim i1 = 0; i1 < Outer1; ++i1)
        {
            for (Dim i2 = 0; i2 < Outer2; ++i2)
            {
                // Helper lambdas
                auto get_val = [&](Dim comp, Dim i)
                {
                    if constexpr (direction == 0)
                        return x_vec.value(comp, i, i1, i2);
                    else if constexpr (direction == 1)
                        return x_vec.value(comp, i1, i, i2);
                    else
                        return x_vec.value(comp, i1, i2, i);
                };

                auto set_rhs = [&](Dim comp, Dim i, Real val)
                {
                    if constexpr (direction == 0)
                        rhs_vec.set(comp, i, i1, i2) = val;
                    else if constexpr (direction == 1)
                        rhs_vec.set(comp, i1, i, i2) = val;
                    else
                        rhs_vec.set(comp, i1, i2, i) = val;
                };

                auto get_gamma_val = [&](Dim i)
                {
                    if constexpr (direction == 0)
                        return gamma_field.get(i, i1, i2);
                    else if constexpr (direction == 1)
                        return gamma_field.get(i1, i, i2);
                    else
                        return gamma_field.get(i1, i2, i);
                };

                // --- COMPONENT 1 (Normal) ---
                if (is_known_face<direction>(i1, i2, Comp1))
                {
                    // Known Face -> Identity Matrix row: RHS = x
                    for (Dim i = 0; i < N; ++i)
                        set_rhs(Comp1, i, get_val(Comp1, i));
                }
                else
                {
                    // Unknown -> Apply Matrix Operator
                    for (Dim i = 0; i < N; ++i)
                    {
                        if (i == 0 || i == N - 1) // Boundaries (Identity for Normal comp)
                        {
                            set_rhs(Comp1, i, get_val(Comp1, i));
                        }
                        else // Internal
                        {
                            Real coeff = get_gamma_val(i) / (h * h);
                            Real val = (-coeff) * get_val(Comp1, i - 1) +
                                       (1.0 + 2.0 * coeff) * get_val(Comp1, i) +
                                       (-coeff) * get_val(Comp1, i + 1);
                            set_rhs(Comp1, i, val);
                        }
                    }
                }

                // --- COMPONENT 2 (Tangent 1) ---
                if (is_known_face<direction>(i1, i2, Comp2))
                {
                    for (Dim i = 0; i < N; ++i)
                        set_rhs(Comp2, i, get_val(Comp2, i));
                }
                else
                {
                    for (Dim i = 0; i < N; ++i)
                    {
                        if (i == 0) // Left Boundary (Identity)
                        {
                            set_rhs(Comp2, i, get_val(Comp2, i));
                        }
                        else if (i == N - 1) // Right Boundary (Modified Neumann)
                        {
                            Real coeff = get_gamma_val(i) / (h * h);
                            Real val = (-coeff) * get_val(Comp2, i - 1) +
                                       (1.0 + 3.0 * coeff) * get_val(Comp2, i);
                            set_rhs(Comp2, i, val);
                        }
                        else // Internal
                        {
                            Real coeff = get_gamma_val(i) / (h * h);
                            Real val = (-coeff) * get_val(Comp2, i - 1) +
                                       (1.0 + 2.0 * coeff) * get_val(Comp2, i) +
                                       (-coeff) * get_val(Comp2, i + 1);
                            set_rhs(Comp2, i, val);
                        }
                    }
                }

                // --- COMPONENT 3 (Tangent 2) ---
                if (is_known_face<direction>(i1, i2, Comp3))
                {
                    for (Dim i = 0; i < N; ++i)
                        set_rhs(Comp3, i, get_val(Comp3, i));
                }
                else
                {
                    for (Dim i = 0; i < N; ++i)
                    {
                        if (i == 0) // Left Boundary (Identity)
                        {
                            set_rhs(Comp3, i, get_val(Comp3, i));
                        }
                        else if (i == N - 1) // Right Boundary (Modified Neumann)
                        {
                            Real coeff = get_gamma_val(i) / (h * h);
                            Real val = (-coeff) * get_val(Comp3, i - 1) +
                                       (1.0 + 3.0 * coeff) * get_val(Comp3, i);
                            set_rhs(Comp3, i, val);
                        }
                        else // Internal
                        {
                            Real coeff = get_gamma_val(i) / (h * h);
                            Real val = (-coeff) * get_val(Comp3, i - 1) +
                                       (1.0 + 2.0 * coeff) * get_val(Comp3, i) +
                                       (-coeff) * get_val(Comp3, i + 1);
                            set_rhs(Comp3, i, val);
                        }
                    }
                }
            }
        }
    }
    template <Dim direction>
    bool is_known_face(Dim index_1, Dim index_2, Dim component) const
    {
        if constexpr (direction == 0)
        {
            if (component == 0)
                return (index_1 == 0 || index_2 == 0);
            if (component == 1)
                return (index_1 == Ny - 1 || index_2 == 0);
            if (component == 2)
                return (index_1 == 0 || index_2 == Nz - 1);
        }
        else if constexpr (direction == 1)
        {
            if (component == 0)
                return (index_1 == Nx - 1 || index_2 == 0);
            if (component == 1)
                return (index_1 == 0 || index_2 == 0);
            if (component == 2)
                return (index_1 == 0 || index_2 == Nz - 1);
        }
        else
        { // direction == 2
            if (component == 0)
                return (index_1 == Nx - 1 || index_2 == 0);
            if (component == 1)
                return (index_1 == 0 || index_2 == Ny - 1);
            if (component == 2)
                return (index_1 == 0 || index_2 == 0);
        }
        throw std::invalid_argument("Invalid component");
    }

    template <Dim direction, typename StrideFunc>
    bool handle_known_face(const DimensionsHandlerVector<StrideFunc> &dim_handler, VectorVariable &solution, Dim index_1, Dim index_2, Dim component)
    {
        if (!is_known_face<direction>(index_1, index_2, component))
            return false;

        Dim Comp1 = dim_handler.Comp1;
        Dim Comp2 = dim_handler.Comp2;
        Dim Comp3 = dim_handler.Comp3;
        const Real t_prev = (t - dt < 0.0f) ? 0.0f : t - dt;

        auto update_bc = [&](Dim i, Dim j, Dim k)
        {
            Real x = i * dx, y = j * dy, z = k * dz;
            if(t == 0.0f){
                solution.set(Comp1, i, j, k) = u_boundary.value<0>(x+dx /Real(2.0), y, z, t);
                solution.set(Comp2, i, j, k) = u_boundary.value<1>(x, y+dy/Real(2.0), z, t);
                solution.set(Comp3, i, j, k) = u_boundary.value<2>(x, y, z+dz/Real(2.0), t);
                return;
            }
            solution.set(Comp1, i, j, k) = u_boundary.value<0>(x+dx /Real(2.0), y, z, t) - u_boundary.value<0>(x+dx/Real(2.0), y, z, t_prev);
            solution.set(Comp2, i, j, k) = u_boundary.value<1>(x, y+dy/Real(2.0), z, t) - u_boundary.value<1>(x, y+dy/Real(2.0), z, t_prev);
            solution.set(Comp3, i, j, k) = u_boundary.value<2>(x, y, z+dz/Real(2.0), t) - u_boundary.value<2>(x, y, z+dz/Real(2.0), t_prev);
        };

        if constexpr (direction == 0)
        {
            for (Dim i = 0; i < Nx; ++i)
                update_bc(i, index_1, index_2);
        }
        else if constexpr (direction == 1)
        {
            for (Dim j = 0; j < Ny; ++j)
                update_bc(index_1, j, index_2);
        }
        else
        {
            for (Dim k = 0; k < Nz; ++k)
                update_bc(index_1, index_2, k);
        }

        return true;
    }
    template <Dim direction>
    void apply_bc(VectorVariable &rhs)
    {
#ifdef USE_MPI
        // In parallel mode, only apply BCs at physical boundaries, not partition interfaces
        bool has_left_boundary = (decomp == nullptr) || decomp->owns_physical_boundary(direction, BoundarySide::LEFT);
        bool has_right_boundary = (decomp == nullptr) || decomp->owns_physical_boundary(direction, BoundarySide::RIGHT);
#else
        bool has_left_boundary = true;
        bool has_right_boundary = true;
#endif

        // Domain lengths:
        Real Lx = dx * (Nx - 0.5);
        Real Ly = dy * (Ny - 0.5);
        Real Lz = dz * (Nz - 0.5);

        if constexpr (direction == 0)
        {
            for (Dim index_1 = 0; index_1 < Ny; ++index_1)
            {
                for (Dim index_2 = 0; index_2 < Nz; ++index_2)
                {
                    // on comp1 we have normal components
                    if (has_left_boundary) {
                        rhs.set(direction, 0, index_1, index_2) = (u_boundary.value<direction>(0, index_1 * dy, index_2 * dz, t) - u_boundary.value<direction>(0, index_1 * dy, index_2 * dz, t - dt)) - ((u_boundary.first_derivative<1>(0, index_1 * dy, index_2 * dz, t, dy) - u_boundary.first_derivative<1>(0, index_1 * dy, index_2 * dz, t - dt, dy)) + (u_boundary.first_derivative<2>(0, index_1 * dy, index_2 * dz, t, dz) - u_boundary.first_derivative<2>(0, index_1 * dy, index_2 * dz, t - dt, dz))) * dx * Real(0.5);
                    }
                    if (has_right_boundary) {
                        rhs.set(direction, Nx - 1, index_1, index_2) = u_boundary.value<direction>(Lx, index_1 * dy, index_2 * dz, t) - u_boundary.value<direction>(Lx, index_1 * dy, index_2 * dz, t - dt);
                    }

                    // on comp2 we have tangent components
                    if (has_left_boundary) {
                        rhs.set(1, 0, index_1, index_2) = u_boundary.value<1>(0, 0.5 * dy + index_1 * dy, index_2 * dz, t) - u_boundary.value<1>(0, 0.5 * dy + index_1 * dy, index_2 * dz, t - dt);
                    }
                    if (has_right_boundary) {
                        rhs.set(1, Nx - 1, index_1, index_2) = rhs.value(1, Nx - 1, index_1, index_2) + Real(2.0) * gamma_field.get(Nx - 1, index_1, index_2) / (dx * dx) * (u_boundary.value<1>(Lx, 0.5 * dy + index_1 * dy, index_2 * dz, t) - u_boundary.value<1>(Lx, 0.5 * dy + index_1 * dy, index_2 * dz, t - dt));
                    }

                    // on comp3 we have tangent components
                    if (has_left_boundary) {
                        rhs.set(2, 0, index_1, index_2) = u_boundary.value<2>(0, index_1 * dy, 0.5 * dz + index_2 * dz, t) - u_boundary.value<2>(0, index_1 * dy, 0.5 * dz + index_2 * dz, t - dt);
                    }
                    if (has_right_boundary) {
                        rhs.set(2, Nx - 1, index_1, index_2) = rhs.value(2, Nx - 1, index_1, index_2) + Real(2.0) * gamma_field.get(Nx - 1, index_1, index_2) / (dx * dx) * (u_boundary.value<2>(Lx, index_1 * dy, 0.5 * dz + index_2 * dz, t) - u_boundary.value<2>(Lx, index_1 * dy, 0.5 * dz + index_2 * dz, t - dt));
                    }
                }
            }
        }
        else if constexpr (direction == 1)
        {
            for (Dim index_1 = 0; index_1 < Nx; ++index_1)
            {
                for (Dim index_2 = 0; index_2 < Nz; ++index_2)
                {
                    // on comp2 we have normal components
                    if (has_left_boundary) {
                        rhs.set(direction, index_1, 0, index_2) = (u_boundary.value<direction>(index_1 * dx, 0, index_2 * dz, t) - u_boundary.value<direction>(index_1 * dx, 0, index_2 * dz, t - dt)) - ((u_boundary.first_derivative<0>(index_1 * dx, 0, index_2 * dz, t, dx) - u_boundary.first_derivative<0>(index_1 * dx, 0, index_2 * dz, t - dt, dx)) + (u_boundary.first_derivative<2>(index_1 * dx, 0, index_2 * dz, t, dz) - u_boundary.first_derivative<2>(index_1 * dx, 0, index_2 * dz, t - dt, dz))) * dy * Real(0.5);
                    }
                    if (has_right_boundary) {
                        rhs.set(direction, index_1, Ny - 1, index_2) = u_boundary.value<direction>(index_1 * dx, Ly, index_2 * dz, t) - u_boundary.value<direction>(index_1 * dx, Ly, index_2 * dz, t - dt);
                    }

                    // on comp1 we have tangent components
                    if (has_left_boundary) {
                        rhs.set(0, index_1, 0, index_2) = u_boundary.value<0>(0.5 * dx + index_1 * dx, 0, index_2 * dz, t) - u_boundary.value<0>(0.5 * dx + index_1 * dx, 0, index_2 * dz, t - dt);
                    }
                    if (has_right_boundary) {
                        rhs.set(0, index_1, Ny - 1, index_2) = rhs.value(0, index_1, Ny - 1, index_2) + Real(2.0) * gamma_field.get(index_1, Ny - 1, index_2) / (dy * dy) * (u_boundary.value<0>(0.5 * dx + index_1 * dx, Ly, index_2 * dz, t) - u_boundary.value<0>(0.5 * dx + index_1 * dx, Ly, index_2 * dz, t - dt));
                    }
                    // on comp3 we have tangent components
                    if (has_left_boundary) {
                        rhs.set(2, index_1, 0, index_2) = u_boundary.value<2>(index_1 * dx, 0, 0.5 * dz + index_2 * dz, t) - u_boundary.value<2>(index_1 * dx, 0, 0.5 * dz + index_2 * dz, t - dt);
                    }
                    if (has_right_boundary) {
                        rhs.set(2, index_1, Ny - 1, index_2) = rhs.value(2, index_1, Ny - 1, index_2) + Real(2.0) * gamma_field.get(index_1, Ny - 1, index_2) / (dy * dy) * (u_boundary.value<2>(index_1 * dx, Ly, 0.5 * dz + index_2 * dz, t) - u_boundary.value<2>(index_1 * dx, Ly, 0.5 * dz + index_2 * dz, t - dt));
                    }
                }
            }
        }
        else if constexpr (direction == 2)
        {
            for (Dim index_1 = 0; index_1 < Nx; ++index_1)
            {
                for (Dim index_2 = 0; index_2 < Ny; ++index_2)
                {
                    // on comp3 we have normal components
                    if (has_left_boundary) {
                        rhs.set(direction, index_1, index_2, 0) = (u_boundary.value<direction>(index_1 * dx, index_2 * dy, 0, t) - u_boundary.value<direction>(index_1 * dx, index_2 * dy, 0, t - dt)) - ((u_boundary.first_derivative<0>(index_1 * dx, index_2 * dy, 0, t, dx) - u_boundary.first_derivative<0>(index_1 * dx, index_2 * dy, 0, t - dt, dx)) + (u_boundary.first_derivative<1>(index_1 * dx, index_2 * dy, 0, t, dy) - u_boundary.first_derivative<1>(index_1 * dx, index_2 * dy, 0, t - dt, dy))) * dz * Real(0.5);
                    }
                    if (has_right_boundary) {
                        rhs.set(direction, index_1, index_2, Nz - 1) = (u_boundary.value<direction>(index_1 * dx, index_2 * dy, dz + (Nz - 1) * dz, t) - u_boundary.value<direction>(index_1 * dx, index_2 * dy, dz + (Nz - 1) * dz, t - dt));
                    }

                    // on comp1 we have tangent components
                    if (has_left_boundary) {
                        rhs.set(0, index_1, index_2, 0) = (u_boundary.value<0>(0.5 * dx + index_1 * dx, index_2 * dy, 0, t) - u_boundary.value<0>(0.5 * dx + index_1 * dx, index_2 * dy, 0, t - dt));
                    }
                    if (has_right_boundary) {
                        rhs.set(0, index_1, index_2, Nz - 1) = rhs.value(0, index_1, index_2, Nz - 1) + Real(2.0) * gamma_field.get(index_1, index_2, Nz - 1) / (dz * dz) * (u_boundary.value<0>(0.5 * dx + index_1 * dx, index_2 * dy, Lz, t) - u_boundary.value<0>(0.5 * dx + index_1 * dx, index_2 * dy, Lz, t - dt));
                    }

                    // on comp2 we have tangent components
                    if (has_left_boundary) {
                        rhs.set(1, index_1, index_2, 0) = (u_boundary.value<1>(index_1 * dx, 0.5 * dy + index_2 * dy, 0, t) - u_boundary.value<1>(index_1 * dx, 0.5 * dy + index_2 * dy, 0, t - dt));
                    }
                    if (has_right_boundary) {
                        rhs.set(1, index_1, index_2, Nz - 1) = rhs.value(1, index_1, index_2, Nz - 1) + Real(2.0) * gamma_field.get(index_1, index_2, Nz - 1) / (dz * dz) * (u_boundary.value<1>(index_1 * dx, 0.5 * dy + index_2 * dy, Lz, t) - u_boundary.value<1>(index_1 * dx, 0.5 * dy + index_2 * dy, Lz, t - dt));
                    }
                }
            }
        }
    };
    template <Dim direction, typename StrideFunc>
    void block_solver(const VectorVariable &rhs, VectorVariable &solution, const DimensionsHandlerVector<StrideFunc> &dim_handler)
    {
        Dim N = (direction == 0) ? Nx : ((direction == 1) ? Ny : Nz);
        Real h = (direction == 0) ? dx : ((direction == 1) ? dy : dz);

        Dim Comp1 = dim_handler.Comp1;
        Dim Comp2 = dim_handler.Comp2;
        Dim Comp3 = dim_handler.Comp3;

        Dim Outer1 = (direction == 0) ? Ny : ((direction == 1) ? Nx : Nx);
        Dim Outer2 = (direction == 0) ? Nz : ((direction == 1) ? Nz : Ny);

#ifdef USE_MPI
        // Batched MPI approach: collect all systems, solve with 3 MPI_Allreduce calls (one per component)
        Dim total_lines = Outer1 * Outer2;
        
        struct LineData {
            Dim i1, i2;
            std::vector<Real> a, b, c;
        };
        
        std::vector<LineData> lines_comp1, lines_comp2, lines_comp3;
        std::vector<std::vector<Real>> rhs_batch_comp1, rhs_batch_comp2, rhs_batch_comp3;
        
        // Prepare data for each component
        for (Dim i1 = 0; i1 < Outer1; ++i1) {
            for (Dim i2 = 0; i2 < Outer2; ++i2) {
                std::vector<Real> a(N), b(N), c(N), d(N);
                
                auto get_gamma = [&](Dim i) {
                    if constexpr (direction == 0)
                        return gamma_field.get(i, i1, i2);
                    else if constexpr (direction == 1)
                        return gamma_field.get(i1, i, i2);
                    else
                        return gamma_field.get(i1, i2, i);
                };
                auto get_rhs_comp = [&](Dim comp, Dim i) {
                    if constexpr (direction == 0)
                        return rhs.value(comp, i, i1, i2);
                    else if constexpr (direction == 1)
                        return rhs.value(comp, i1, i, i2);
                    else
                        return rhs.value(comp, i1, i2, i);
                };
                
                // Comp1 (Normal) - if not handled by boundary
                if (!handle_known_face<direction>(dim_handler, solution, i1, i2, Comp1)) {
                    setup_TDMA_internal(N, h, a, b, c, d, [&](Dim i) { return get_rhs_comp(Comp1, i); }, get_gamma);
                    a[0] = 0.0; b[0] = 1.0; c[0] = 0.0; d[0] = get_rhs_comp(Comp1, 0);
                    a[N-1] = 0.0; b[N-1] = 1.0; c[N-1] = 0.0; d[N-1] = get_rhs_comp(Comp1, N-1);
                    lines_comp1.push_back({i1, i2, a, b, c});
                    rhs_batch_comp1.push_back(d);
                }
                
                // Comp2 (Tangent)
                if (!handle_known_face<direction>(dim_handler, solution, i1, i2, Comp2)) {
                    setup_TDMA_internal(N, h, a, b, c, d, [&](Dim i) { return get_rhs_comp(Comp2, i); }, get_gamma);
                    a[0] = 0.0; b[0] = 1.0; c[0] = 0.0; d[0] = get_rhs_comp(Comp2, 0);
                    Real gamma_N = get_gamma(N - 1);
                    Real coeff = gamma_N / (h * h);
                    a[N-1] = -coeff; b[N-1] = (1.0f + 2.0f * coeff) - (-coeff); c[N-1] = 0.0;
                    d[N-1] = get_rhs_comp(Comp2, N-1);
                    lines_comp2.push_back({i1, i2, a, b, c});
                    rhs_batch_comp2.push_back(d);
                }
                
                // Comp3 (Tangent)
                if (!handle_known_face<direction>(dim_handler, solution, i1, i2, Comp3)) {
                    setup_TDMA_internal(N, h, a, b, c, d, [&](Dim i) { return get_rhs_comp(Comp3, i); }, get_gamma);
                    a[0] = 0.0; b[0] = 1.0; c[0] = 0.0; d[0] = get_rhs_comp(Comp3, 0);
                    Real gamma_N = get_gamma(N - 1);
                    Real coeff = gamma_N / (h * h);
                    a[N-1] = -coeff; b[N-1] = (1.0f + 2.0f * coeff) - (-coeff); c[N-1] = 0.0;
                    d[N-1] = get_rhs_comp(Comp3, N-1);
                    lines_comp3.push_back({i1, i2, a, b, c});
                    rhs_batch_comp3.push_back(d);
                }
            }
        }
        
        // Batch solve for each component (3 MPI_Allreduce calls total instead of 3*Outer1*Outer2)
        auto solve_and_write = [&](const std::vector<LineData>& lines, 
                                   const std::vector<std::vector<Real>>& rhs_batch,
                                   Dim comp) {
            if (lines.empty()) return;
            
            std::vector<std::vector<Real>> a_batch, b_batch, c_batch, x_batch(lines.size(), std::vector<Real>(N));
            for (const auto& line : lines) {
                a_batch.push_back(line.a);
                b_batch.push_back(line.b);
                c_batch.push_back(line.c);
            }
            
            auto rhs_batch_copy = rhs_batch; // Need mutable copy
            batched_schur_complement_solver(a_batch, b_batch, c_batch, rhs_batch_copy, x_batch, direction);
            
            // Write solutions back
            for (size_t idx = 0; idx < lines.size(); ++idx) {
                Dim i1 = lines[idx].i1;
                Dim i2 = lines[idx].i2;
                for (Dim i = 0; i < N; ++i) {
                    if constexpr (direction == 0)
                        solution.set(comp, i, i1, i2) = x_batch[idx][i];
                    else if constexpr (direction == 1)
                        solution.set(comp, i1, i, i2) = x_batch[idx][i];
                    else
                        solution.set(comp, i1, i2, i) = x_batch[idx][i];
                }
            }
        };
        
        solve_and_write(lines_comp1, rhs_batch_comp1, Comp1);
        solve_and_write(lines_comp2, rhs_batch_comp2, Comp2);
        solve_and_write(lines_comp3, rhs_batch_comp3, Comp3);
        
#else
        // Serial version: solve line by line
        std::vector<Real> a(N), b(N), c(N), d(N), x(N);
        
        for (Dim i1 = 0; i1 < Outer1; ++i1)
        {
            for (Dim i2 = 0; i2 < Outer2; ++i2)
            {
                auto get_gamma = [&](Dim i)
                {
                    if constexpr (direction == 0)
                        return gamma_field.get(i, i1, i2);
                    else if constexpr (direction == 1)
                        return gamma_field.get(i1, i, i2);
                    else
                        return gamma_field.get(i1, i2, i);
                };
                auto get_rhs_comp = [&](Dim comp, Dim i)
                {
                    if constexpr (direction == 0)
                        return rhs.value(comp, i, i1, i2);
                    else if constexpr (direction == 1)
                        return rhs.value(comp, i1, i, i2);
                    else
                        return rhs.value(comp, i1, i2, i);
                };
                auto set_sol_comp = [&](Dim comp, Dim i, Real val)
                {
                    if constexpr (direction == 0)
                        solution.set(comp, i, i1, i2) = val;
                    else if constexpr (direction == 1)
                        solution.set(comp, i1, i, i2) = val;
                    else
                        solution.set(comp, i1, i2, i) = val;
                };

                // Comp1 (Normal)
                if (!handle_known_face<direction>(dim_handler, solution, i1, i2, Comp1))
                {
                    setup_TDMA_internal(N, h, a, b, c, d, [&](Dim i)
                                        { return get_rhs_comp(Comp1, i); }, get_gamma);
                    a[0] = 0.0;
                    b[0] = 1.0;
                    c[0] = 0.0;
                    d[0] = get_rhs_comp(Comp1, 0);
                    a[N - 1] = 0.0;
                    b[N - 1] = 1.0;
                    c[N - 1] = 0.0;
                    d[N - 1] = get_rhs_comp(Comp1, N - 1);
                    thomas_algorithm(a, b, c, d, x);
                    for (Dim i = 0; i < N; ++i)
                        set_sol_comp(Comp1, i, x[i]);
                }

                // Comp2 (Tangent)
                if (!handle_known_face<direction>(dim_handler, solution, i1, i2, Comp2))
                {
                    setup_TDMA_internal(N, h, a, b, c, d, [&](Dim i)
                                        { return get_rhs_comp(Comp2, i); }, get_gamma);
                    a[0] = 0.0;
                    b[0] = 1.0;
                    c[0] = 0.0;
                    d[0] = get_rhs_comp(Comp2, 0);
                    Real gamma_N = get_gamma(N - 1);
                    Real coeff = gamma_N / (h * h);
                    a[N - 1] = -coeff;
                    b[N - 1] = (1.0f + 2.0f * coeff) - (-coeff);
                    c[N - 1] = 0.0;
                    d[N - 1] = get_rhs_comp(Comp2, N - 1);
                    thomas_algorithm(a, b, c, d, x);
                    for (Dim i = 0; i < N; ++i)
                        set_sol_comp(Comp2, i, x[i]);
                }

                // Comp3 (Tangent)
                if (!handle_known_face<direction>(dim_handler, solution, i1, i2, Comp3))
                {
                    setup_TDMA_internal(N, h, a, b, c, d, [&](Dim i)
                                        { return get_rhs_comp(Comp3, i); }, get_gamma);
                    a[0] = 0.0;
                    b[0] = 1.0;
                    c[0] = 0.0;
                    d[0] = get_rhs_comp(Comp3, 0);
                    Real gamma_N = get_gamma(N - 1);
                    Real coeff = gamma_N / (h * h);
                    a[N - 1] = -coeff;
                    b[N - 1] = (1.0f + 2.0f * coeff) - (-coeff);
                    c[N - 1] = 0.0;
                    d[N - 1] = get_rhs_comp(Comp3, N - 1);
                    thomas_algorithm(a, b, c, d, x);
                    for (Dim i = 0; i < N; ++i)
                        set_sol_comp(Comp3, i, x[i]);
                }
            }
        }
#endif
    }

    VelocitySolver(Dim Nx_, Dim Ny_, Dim Nz_, Real dx_, Real dy_, Real dz_, Real dt_, ScalarVariable &gam, BoundaryFunctions &u_bnd)
        : Solver(Nx_, Ny_, Nz_, dx_, dy_, dz_, dt_), gamma_field(gam), u_boundary(u_bnd) {}

    template <typename StrideFunc, Dim direction>
    void solve(VectorVariable &rhs, VectorVariable &solution, const DimensionsHandlerVector<StrideFunc> &dim_handler)
    {
        apply_bc<direction>(rhs);
        block_solver<direction, StrideFunc>(rhs, solution, dim_handler);
        advance_time();

    };
    void set_gamma(ScalarVariable &g) { gamma_field = g; }
    BoundaryFunctions &set_u_boundary() { return u_boundary; }
};
#endif // SOLVER_HPP
