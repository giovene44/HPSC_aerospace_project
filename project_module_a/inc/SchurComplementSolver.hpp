#ifndef SCHUR_COMPLEMENT_SOLVER_HPP
#define SCHUR_COMPLEMENT_SOLVER_HPP

#include "SchurMatrixData.hpp"
#include "MPICommunicator.hpp"
#include <vector>
#include <cmath>
#include <stdexcept>

struct LineCache
{
    bool has_internal = false;
    Real y0 = Real(0);
    Real yLast = Real(0);
};

struct ThreadWorkspace
{
    std::vector<Real> internal_rhs;
    std::vector<Real> y;
    std::vector<Real> g;
    std::vector<Real> u_i;
    std::vector<Real> iface_rhs;
    std::vector<Real> iface_sol;
};

/**
 * @brief Parallel tridiagonal solver using Schur complement method.
 *
 * Implements the algorithm from Lecture 5 slides 26-32:
 *
 * 1. Reorder unknowns into internal (u_i) and interface (u_s) points
 * 2. Block form: [A_ii A_is; A_si A_ss] [u_i; u_s] = [f_i; f_s]
 * 3. Schur complement: S = A_ss - A_si * A_ii^(-1) * A_is (tridiagonal!)
 *
 * Algorithm phases:
 * - Preprocessing (once): Compute Schur complement matrix S
 * - Runtime solve (each timestep):
 *   a. Local: Solve y = A_ii^(-1) * f_i
 *   b. Local: Compute interface RHS contribution
 *   c. MPI: Gather interface RHS contributions
 *   d. Root: Solve reduced system S * u_s = f_s_modified
 *   e. MPI: Broadcast interface solution
 *   f. Local: Back-substitute u_i = A_ii^(-1) * (f_i - A_is * u_s)
 *
 * @note This solver handles a single 1D tridiagonal system distributed across
 *       multiple MPI processes. For 3D problems, use one solver per direction
 *       and apply to independent 1D lines.
 */
class SchurComplementSolver
{
public:
    /**
     * @brief Constructor with domain decomposition parameters.
     * @param global_N Global problem size in solve direction
     * @param num_procs Number of processes in solve direction
     * @param my_rank This process's rank in solve direction (0 to num_procs-1)
     * @param comm MPI communicator wrapper
     */
    SchurComplementSolver(Dim global_N, int num_procs, int my_rank, MPICommunicator &comm);

    /**
     * @brief Preprocessing phase: compute Schur complement contributions.
     *
     * Must be called before solve() and whenever coefficients change.
     * For constant-coefficient problems, call once at setup.
     *
     * @param a Lower diagonal coefficients (size: local_N)
     * @param b Main diagonal coefficients (size: local_N)
     * @param c Upper diagonal coefficients (size: local_N)
     */
    void preprocess(const std::vector<Real> &a,
                    const std::vector<Real> &b,
                    const std::vector<Real> &c);

    /**
     * @brief Solve tridiagonal system using Schur complement.
     *
     * @param rhs Right-hand side (size: local_N)
     * @param solution Output solution (size: local_N)
     */
    void solve(const std::vector<Real> &rhs,
               std::vector<Real> &solution);

    /**
     * @brief Check if preprocessing has been done.
     */
    bool is_preprocessed() const { return schur_data_.is_preprocessed; }

    /**
     * @brief Get local problem size (including interface overlap).
     */
    Dim get_local_N() const { return schur_data_.local_N; }

    /**
     * @brief Get global starting index for this process.
     */
    Dim get_global_start() const { return schur_data_.global_start; }

    /**
     * @brief Get number of internal points (excluding interfaces).
     */
    Dim get_n_internal() const { return schur_data_.n_internal; }

    /**
     * @brief Get the Schur complement data (for debugging/testing).
     */
    const SchurComplementData &get_schur_data() const { return schur_data_; }

    void compute_local_contrib(
        const Real *rhs_local, int local_N,
        std::vector<Real> &iface_rhs_out, // size = num_interfaces
        LineCache &cache_out,
        ThreadWorkspace &ws) const;

    void allreduce_contribs(
        const std::vector<Real> &local_flat,
        std::vector<Real> &global_flat) const;

    void finalize_line(
        const Real *rhs_local, int local_N,
        const std::vector<Real> &global_iface_rhs, // size = num_interfaces
        Real *sol_local_out,                       // size = local_N
        ThreadWorkspace &ws);

    void solve_batch(
        const Real *rhs_lines_flat, int nLines,
        Real *sol_lines_flat,
        bool use_omp);

private:
    // Reference to MPI communicator
    MPICommunicator &comm_;

    // Schur complement data
    SchurComplementData schur_data_;

    // Cached original tridiagonal coefficients (local portion)
    std::vector<Real> a_, b_, c_;

    // Workspace vectors
    std::vector<Real> work_y_;              // Intermediate solution y = A_ii^(-1) * f_i
    std::vector<Real> work_internal_rhs_;   // Modified RHS for internal solve
    std::vector<Real> local_interface_rhs_; // Local contribution to interface RHS

    // ==================== Helper Methods ====================

    /**
     * @brief Compute columns of A_ii^(-1) * A_is needed for Schur complement.
     *
     * Solves A_ii * x = e_j for j = first and last internal indices.
     */
    void compute_inv_Aii_columns();

    /**
     * @brief Compute local Schur complement diagonal and off-diagonal contributions.
     */
    void compute_local_schur_contributions();

    /**
     * @brief Assemble global Schur complement matrix from all processes.
     */
    void assemble_global_schur_system();

    /**
     * @brief Solve the reduced interface system S * u_s = f_s.
     *
     * Called on root process; uses Thomas algorithm on small system.
     *
     * @param interface_rhs Right-hand side of reduced system
     * @param interface_sol Output: solution at interfaces
     */
    void solve_interface_system(const std::vector<Real> &interface_rhs,
                                std::vector<Real> &interface_sol);

    /**
     * @brief Thomas algorithm for tridiagonal system.
     *
     * Solves: a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = rhs[i]
     *
     * @param a Lower diagonal (a[0] unused)
     * @param b Main diagonal
     * @param c Upper diagonal (c[n-1] unused)
     * @param rhs Right-hand side
     * @param x Output solution
     */
    static void thomas_solve(const std::vector<Real> &a,
                             const std::vector<Real> &b,
                             const std::vector<Real> &c,
                             const std::vector<Real> &rhs,
                             std::vector<Real> &x);

    /**
     * @brief Thomas algorithm for tridiagonal system with modified boundaries.
     *
     * Used for internal block solve where boundary conditions come from interfaces.
     *
     * @param a Lower diagonal
     * @param b Main diagonal
     * @param c Upper diagonal
     * @param rhs Right-hand side
     * @param x Output solution
     * @param start Start index
     * @param end End index (exclusive)
     */
    static void thomas_solve_range(const std::vector<Real> &a,
                                   const std::vector<Real> &b,
                                   const std::vector<Real> &c,
                                   const std::vector<Real> &rhs,
                                   std::vector<Real> &x,
                                   Dim start, Dim end);
};

// ==================== Implementation ====================

inline SchurComplementSolver::SchurComplementSolver(Dim global_N, int num_procs, int my_rank,
                                                    MPICommunicator &comm)
    : comm_(comm)
{
    // Initialize Schur data with topology
    schur_data_.init(global_N, num_procs, my_rank);

    // Allocate workspace
    work_y_.resize(schur_data_.n_internal, 0.0);
    work_internal_rhs_.resize(schur_data_.n_internal, 0.0);

    // Interface RHS has 2 entries per process (left and right interface)
    // But we only contribute to interfaces we have
    local_interface_rhs_.resize(schur_data_.get_num_interfaces(), 0.0);
}

inline void SchurComplementSolver::preprocess(const std::vector<Real> &a,
                                              const std::vector<Real> &b,
                                              const std::vector<Real> &c)
{
    // Validate input sizes
    if (a.size() != schur_data_.local_N || b.size() != schur_data_.local_N ||
        c.size() != schur_data_.local_N)
    {
        throw std::runtime_error("SchurComplementSolver::preprocess: coefficient size mismatch");
    }

    // Store coefficients
    a_ = a;
    b_ = b;
    c_ = c;

    // Store interface coupling coefficients
    SchurBlockData &block = schur_data_.block_data;

    if (schur_data_.has_left_interface)
    {
        block.b_left_interface = b[0];
        block.a_left_interface = a[0]; // This couples to left neighbor (unused in internal)
        block.c_left_interface = c[0]; // This couples left interface to first internal
    }

    if (schur_data_.has_right_interface)
    {
        Dim ri = schur_data_.local_N - 1;
        block.b_right_interface = b[ri];
        block.a_right_interface = a[ri]; // This couples right interface to last internal
        block.c_right_interface = c[ri]; // This couples to right neighbor (unused in internal)
    }

    // Extract internal block coefficients
    block.n_internal = schur_data_.n_internal;
    if (block.n_internal > 0)
    {
        block.a_internal.resize(block.n_internal);
        block.b_internal.resize(block.n_internal);
        block.c_internal.resize(block.n_internal);

        // Internal indices depend on which interfaces we have
        Dim internal_start = schur_data_.has_left_interface ? 1 : 0;
        for (Dim i = 0; i < block.n_internal; ++i)
        {
            block.a_internal[i] = a[internal_start + i];
            block.b_internal[i] = b[internal_start + i];
            block.c_internal[i] = c[internal_start + i];
        }
    }

    // Compute A_ii^(-1) columns for Schur complement
    compute_inv_Aii_columns();

    // Compute local Schur complement contributions
    compute_local_schur_contributions();

    // Assemble global Schur system via MPI
    assemble_global_schur_system();

    schur_data_.is_preprocessed = true;
}

inline void SchurComplementSolver::compute_inv_Aii_columns()
{
    SchurBlockData &block = schur_data_.block_data;

    if (block.n_internal == 0)
    {
        // No internal points - special handling for very small blocks
        block.inv_Aii_col_left.clear();
        block.inv_Aii_col_right.clear();
        return;
    }

    block.inv_Aii_col_left.resize(block.n_internal, 0.0);
    block.inv_Aii_col_right.resize(block.n_internal, 0.0);

    // We need to solve A_ii * col = e for specific right-hand sides
    // that represent the coupling from interfaces to internal points.

    // For left interface coupling: A_is column (internal eqs, interface unknown) = [a[1], 0, ...]^T
    // We need A_ii^(-1) * A_is = A_ii^(-1) * [a_internal[0], 0, ...]^T
    if (schur_data_.has_left_interface && block.n_internal > 0)
    {
        std::vector<Real> rhs_left(block.n_internal, 0.0);
        rhs_left[0] = block.a_internal[0]; // a[1] couples first internal eq to left interface

        thomas_solve(block.a_internal, block.b_internal, block.c_internal,
                     rhs_left, block.inv_Aii_col_left);
    }

    // For right interface coupling: A_is column = [0, ..., c[N-1]]^T
    // We need A_ii^(-1) * [0, ..., c_internal[n-1]]^T
    if (schur_data_.has_right_interface && block.n_internal > 0)
    {
        std::vector<Real> rhs_right(block.n_internal, 0.0);
        rhs_right[block.n_internal - 1] = block.c_internal[block.n_internal - 1];

        thomas_solve(block.a_internal, block.b_internal, block.c_internal,
                     rhs_right, block.inv_Aii_col_right);
    }
}

inline void SchurComplementSolver::compute_local_schur_contributions()
{
    SchurBlockData &block = schur_data_.block_data;

    // Schur complement: S = A_ss - A_si * A_ii^(-1) * A_is
    //
    // A_ss is diagonal at interfaces: A_ss[left] = b[0], A_ss[right] = b[N-1]
    //
    // A_si * A_ii^(-1) * A_is gives the fill-in from eliminating internal unknowns
    // A_si row (interface eq): c[0] couples left interface to first internal
    //                          a[N-1] couples right interface to last internal
    // A_is column (from internal eqs): a[1] couples first internal to left interface
    //                                  c[N-1] couples last internal to right interface
    //
    // So the Schur diagonal contribution:
    // S[left,left] = b[0] - c[0] * (A_ii^(-1) * [a[1], 0, ...])[0]
    //              = b[0] - c[0] * inv_Aii_col_left[0]
    // S[right,right] = b[N-1] - a[N-1] * (A_ii^(-1) * [0, ..., c[N-1]])[n-1]
    //                = b[N-1] - a[N-1] * inv_Aii_col_right[n-1]

    // For the Schur complement diagonal, we need: S = A_ss - A_si * A_ii^(-1) * A_is
    // The b[interface] (A_ss) should only be counted once per interface, not by both processes.
    // We'll have the "left" process (the one with right_interface) contribute the full S,
    // and the "right" process (with left_interface) only contribute the fill-in correction.

    // Initialize to zero
    block.schur_diag_left = 0.0;
    block.schur_diag_right = 0.0;
    block.schur_offdiag_to_left = 0.0;
    block.schur_offdiag_to_right = 0.0;

    if (schur_data_.has_left_interface)
    {
        if (block.n_internal > 0)
        {
            // This process has a LEFT interface - only contribute the fill-in correction
            // The b[interface] will be added by the other process (which has right_interface to this point)
            block.schur_diag_left = -block.c_left_interface * block.inv_Aii_col_left[0];

            // Off-diagonal: coupling from left interface to right interface (S_local[0,1])
            // This is used for S[i, i+1] where i is the left interface's global index
            if (schur_data_.has_right_interface)
            {
                // Only compute if we have both interfaces (i.e., internal unknowns between them)
                block.schur_offdiag_to_right = -block.c_left_interface * block.inv_Aii_col_right[0];
            }
        }
    }

    if (schur_data_.has_right_interface)
    {
        if (block.n_internal > 0)
        {
            // This process has a RIGHT interface - contribute b[interface] plus fill-in correction
            block.schur_diag_right = block.b_right_interface -
                                     block.a_right_interface * block.inv_Aii_col_right[block.n_internal - 1];

            // Off-diagonal: coupling from right interface to left interface (S_local[1,0])
            // This is used for S[i, i-1] where i is the right interface's global index
            if (schur_data_.has_left_interface)
            {
                // Only compute if we have both interfaces (i.e., internal unknowns between them)
                block.schur_offdiag_to_left = -block.a_right_interface * block.inv_Aii_col_left[block.n_internal - 1];
            }
        }
        else
        {
            block.schur_diag_right = block.b_right_interface;
        }
    }
}

inline void SchurComplementSolver::assemble_global_schur_system()
{
    int num_interfaces = schur_data_.get_num_interfaces();
    if (num_interfaces == 0)
    {
        // Single process, no interfaces to worry about
        return;
    }

    // Each interface is shared between two adjacent processes.
    // Process p owns the interface between processes p and p+1.
    // This interface's Schur diagonal is the sum of:
    //   - Process p's right interface contribution (schur_diag_right)
    //   - Process p+1's left interface contribution (schur_diag_left)

    // Gather all contributions
    // Local contribution: [schur_diag_left, schur_diag_right, schur_offdiag_to_left, schur_offdiag_to_right]
    std::vector<Real> local_contrib(4, 0.0);
    local_contrib[0] = schur_data_.has_left_interface ? schur_data_.block_data.schur_diag_left : 0.0;
    local_contrib[1] = schur_data_.has_right_interface ? schur_data_.block_data.schur_diag_right : 0.0;
    local_contrib[2] = schur_data_.has_left_interface ? schur_data_.block_data.schur_offdiag_to_left : 0.0;
    local_contrib[3] = schur_data_.has_right_interface ? schur_data_.block_data.schur_offdiag_to_right : 0.0;

    std::vector<Real> all_contrib;
    comm_.allgather_schur_diag(local_contrib, all_contrib);

    // Assemble the tridiagonal Schur matrix S
    // Interface i (between processes i and i+1):
    //   S[i,i] = all_contrib[4*i + 1] + all_contrib[4*(i+1) + 0]
    //          = process i's right diag + process (i+1)'s left diag
    //   S[i,i-1] = all_contrib[4*i + 2]  (offdiag to left from process i)
    //   S[i,i+1] = all_contrib[4*i + 3]  (offdiag to right from process i)

    schur_data_.schur_a.resize(num_interfaces, 0.0);
    schur_data_.schur_b.resize(num_interfaces, 0.0);
    schur_data_.schur_c.resize(num_interfaces, 0.0);

    for (int i = 0; i < num_interfaces; ++i)
    {
        // Diagonal: sum of contributions from processes i (right) and i+1 (left)
        Real diag_from_left_proc = all_contrib[4 * i + 1];        // Process i's right interface (schur_diag_right)
        Real diag_from_right_proc = all_contrib[4 * (i + 1) + 0]; // Process i+1's left interface (schur_diag_left)
        schur_data_.schur_b[i] = diag_from_left_proc + diag_from_right_proc;

        // Lower diagonal: coupling from interface i to interface i-1
        // This comes from process i which has both interface i-1 (left) and interface i (right)
        // Process i's S_local[1,0] = offdiag_to_left
        if (i > 0)
        {
            schur_data_.schur_a[i] = all_contrib[4 * i + 2]; // Process i's offdiag_to_left
        }

        // Upper diagonal: coupling from interface i to interface i+1
        // This comes from process i+1 which has both interface i (left) and interface i+1 (right)
        // Process i+1's S_local[0,1] = offdiag_to_right
        if (i < num_interfaces - 1)
        {
            schur_data_.schur_c[i] = all_contrib[4 * (i + 1) + 3]; // Process i+1's offdiag_to_right
        }
    }
}

inline void SchurComplementSolver::solve(const std::vector<Real> &rhs, std::vector<Real> &solution)
{
    if (!schur_data_.is_preprocessed)
    {
        throw std::runtime_error("SchurComplementSolver::solve: must call preprocess() first");
    }

    if (rhs.size() != schur_data_.local_N)
    {
        throw std::runtime_error("SchurComplementSolver::solve: rhs size mismatch");
    }

    solution.resize(schur_data_.local_N);
    const SchurBlockData &block = schur_data_.block_data;
    int num_interfaces = schur_data_.get_num_interfaces();

    // ========== Phase 1: Solve y = A_ii^(-1) * f_i ==========
    // Extract internal RHS and solve local tridiagonal system

    if (block.n_internal > 0)
    {
        Dim internal_start = schur_data_.has_left_interface ? 1 : 0;

        // Copy internal RHS
        work_internal_rhs_.resize(block.n_internal);
        for (Dim i = 0; i < block.n_internal; ++i)
        {
            work_internal_rhs_[i] = rhs[internal_start + i];
        }

        // Solve A_ii * y = f_i
        work_y_.resize(block.n_internal);
        thomas_solve(block.a_internal, block.b_internal, block.c_internal,
                     work_internal_rhs_, work_y_);
    }

    // ========== Phase 2: Compute interface RHS contribution ==========
    // f_s_modified = f_s - A_si * y
    // A_si row (from interface eqs): c[0] couples left interface eq to first internal unknown
    //                                a[N-1] couples right interface eq to last internal unknown
    // So contribution: f_s[left] -= c[0] * y[0]
    //                  f_s[right] -= a[N-1] * y[n-1]

    std::fill(local_interface_rhs_.begin(), local_interface_rhs_.end(), 0.0);

    if (num_interfaces > 0)
    {
        // Left interface contribution
        // Similar to Schur diagonal: process with left_interface only contributes the fill-in,
        // while process with right_interface contributes f_s + fill-in
        if (schur_data_.has_left_interface)
        {
            int iface_idx = schur_data_.get_interface_index(0); // Left interface index
            if (iface_idx >= 0 && iface_idx < num_interfaces)
            {
                Real contribution = 0.0; // Don't include f_s[left] - other process will add it
                if (block.n_internal > 0)
                {
                    // Subtract A_si * y contribution
                    // A_si row for left interface: c[0] couples to first internal
                    contribution -= block.c_left_interface * work_y_[0];
                }
                local_interface_rhs_[iface_idx] += contribution;
            }
        }

        // Right interface contribution
        if (schur_data_.has_right_interface)
        {
            int iface_idx = schur_data_.get_interface_index(1); // Right interface index
            if (iface_idx >= 0 && iface_idx < num_interfaces)
            {
                Real contribution = rhs[schur_data_.local_N - 1]; // f_s[right] - include once
                if (block.n_internal > 0)
                {
                    // Subtract A_si * y contribution
                    // A_si row for right interface: a[N-1] couples to last internal
                    contribution -= block.a_right_interface * work_y_[block.n_internal - 1];
                }
                local_interface_rhs_[iface_idx] += contribution;
            }
        }
    }

    // ========== Phase 3: Gather interface RHS and solve reduced system ==========
    // Use Allreduce instead of Reduce+Bcast: all processes get the sum and solve locally.
    // The interface system is tiny (size = num_procs - 1), so redundant solve is cheaper
    // than the extra collective communication.
    std::vector<Real> global_interface_rhs(num_interfaces, 0.0);
    if (num_interfaces > 0)
    {
        comm_.allreduce_interface_rhs(local_interface_rhs_, global_interface_rhs);

        // All processes solve the (tiny) reduced system locally
        solve_interface_system(global_interface_rhs, schur_data_.interface_solution);
    }

    // ========== Phase 4: Back-substitute for internal solution ==========
    // u_i = A_ii^(-1) * (f_i - A_is * u_s)
    // A_is has: c[0] at (left_interface, first_internal)
    //           a[N-1] at (right_interface, last_internal)

    if (block.n_internal > 0)
    {
        Dim internal_start = schur_data_.has_left_interface ? 1 : 0;

        // Modify RHS: g = f_i - A_is * u_s
        std::vector<Real> g(block.n_internal);
        for (Dim i = 0; i < block.n_internal; ++i)
        {
            g[i] = rhs[internal_start + i];
        }

        // Subtract interface contributions (A_is * u_s)
        // A_is column for left interface: [a_internal[0], 0, ...]^T (couples interface to first internal eq)
        // A_is column for right interface: [0, ..., c_internal[n-1]]^T (couples interface to last internal eq)
        if (schur_data_.has_left_interface)
        {
            int left_iface_idx = schur_data_.get_interface_index(0);
            if (left_iface_idx >= 0)
            {
                Real u_s_left = schur_data_.interface_solution[left_iface_idx];
                g[0] -= block.a_internal[0] * u_s_left;
            }
        }

        if (schur_data_.has_right_interface)
        {
            int right_iface_idx = schur_data_.get_interface_index(1);
            if (right_iface_idx >= 0)
            {
                Real u_s_right = schur_data_.interface_solution[right_iface_idx];
                g[block.n_internal - 1] -= block.c_internal[block.n_internal - 1] * u_s_right;
            }
        }

        // Solve A_ii * u_i = g
        std::vector<Real> u_i(block.n_internal);
        thomas_solve(block.a_internal, block.b_internal, block.c_internal, g, u_i);

        // Copy internal solution
        for (Dim i = 0; i < block.n_internal; ++i)
        {
            solution[internal_start + i] = u_i[i];
        }
    }

    // ========== Phase 5: Set interface values in solution ==========
    if (schur_data_.has_left_interface)
    {
        int left_iface_idx = schur_data_.get_interface_index(0);
        if (left_iface_idx >= 0)
        {
            solution[0] = schur_data_.interface_solution[left_iface_idx];
        }
    }

    if (schur_data_.has_right_interface)
    {
        int right_iface_idx = schur_data_.get_interface_index(1);
        if (right_iface_idx >= 0)
        {
            solution[schur_data_.local_N - 1] = schur_data_.interface_solution[right_iface_idx];
        }
    }

    // Handle single-process case (no interfaces)
    if (num_interfaces == 0)
    {
        // Just solve the full system directly
        thomas_solve(a_, b_, c_, rhs, solution);
    }
}

inline void SchurComplementSolver::solve_interface_system(const std::vector<Real> &interface_rhs,
                                                          std::vector<Real> &interface_sol)
{
    int n = static_cast<int>(interface_rhs.size());
    if (n == 0)
        return;

    interface_sol.resize(n);

    // Solve S * u_s = f_s using Thomas algorithm
    thomas_solve(schur_data_.schur_a, schur_data_.schur_b, schur_data_.schur_c,
                 interface_rhs, interface_sol);
}

inline void SchurComplementSolver::thomas_solve(const std::vector<Real> &a,
                                                const std::vector<Real> &b,
                                                const std::vector<Real> &c,
                                                const std::vector<Real> &rhs,
                                                std::vector<Real> &x)
{
    Dim n = static_cast<Dim>(b.size());
    if (n == 0)
        return;

    x.resize(n);

    // Create working copies for modified coefficients
    std::vector<Real> c_prime(n);
    std::vector<Real> d_prime(n);

    // Forward sweep
    c_prime[0] = c[0] / b[0];
    d_prime[0] = rhs[0] / b[0];

    for (Dim i = 1; i < n; ++i)
    {
        Real denom = b[i] - a[i] * c_prime[i - 1];
        if (std::abs(denom) < 1e-15)
        {
            throw std::runtime_error("Thomas algorithm: zero pivot encountered");
        }
        c_prime[i] = c[i] / denom;
        d_prime[i] = (rhs[i] - a[i] * d_prime[i - 1]) / denom;
    }

    // Back substitution
    x[n - 1] = d_prime[n - 1];
    for (Dim i = n - 1; i > 0; --i)
    {
        x[i - 1] = d_prime[i - 1] - c_prime[i - 1] * x[i];
    }
}

inline void SchurComplementSolver::thomas_solve_range(const std::vector<Real> &a,
                                                      const std::vector<Real> &b,
                                                      const std::vector<Real> &c,
                                                      const std::vector<Real> &rhs,
                                                      std::vector<Real> &x,
                                                      Dim start, Dim end)
{
    Dim n = end - start;
    if (n == 0)
        return;

    // Create working copies for modified coefficients
    std::vector<Real> c_prime(n);
    std::vector<Real> d_prime(n);

    // Forward sweep
    c_prime[0] = c[start] / b[start];
    d_prime[0] = rhs[start] / b[start];

    for (Dim i = 1; i < n; ++i)
    {
        Real denom = b[start + i] - a[start + i] * c_prime[i - 1];
        if (std::abs(denom) < 1e-15)
        {
            throw std::runtime_error("Thomas algorithm: zero pivot encountered");
        }
        c_prime[i] = c[start + i] / denom;
        d_prime[i] = (rhs[start + i] - a[start + i] * d_prime[i - 1]) / denom;
    }

    // Back substitution
    x[start + n - 1] = d_prime[n - 1];
    for (Dim i = n - 1; i > 0; --i)
    {
        x[start + i - 1] = d_prime[i - 1] - c_prime[i - 1] * x[start + i];
    }
}

// =============================================================
// Thread-safe batched Schur solve
// (Phase 1-2) compute local contributions in parallel
// (Phase 3)   one MPI_Allreduce for all lines
// (Phase 4-5) finalize lines in parallel
// =============================================================

// ---------------------------
// Phase 1-2: NO MPI
// ---------------------------
inline void SchurComplementSolver::compute_local_contrib(
    const Real *rhs_local, int local_N,
    std::vector<Real> &iface_rhs_out, // size = num_interfaces
    LineCache &cache_out,
    ThreadWorkspace &ws) const
{
    if (!schur_data_.is_preprocessed)
    {
        throw std::runtime_error("compute_local_contrib: must preprocess() first");
    }
    if (local_N != schur_data_.local_N)
    {
        throw std::runtime_error("compute_local_contrib: local_N mismatch");
    }

    const SchurBlockData &block = schur_data_.block_data;
    const int num_if = schur_data_.get_num_interfaces();

    iface_rhs_out.assign(num_if, Real(0));

    // ---- Phase 1: solve internal y = A_ii^{-1} f_i ----
    cache_out.has_internal = false;
    cache_out.y0 = cache_out.yLast = Real(0);

    if (block.n_internal > 0)
    {
        const Dim internal_start = schur_data_.has_left_interface ? 1 : 0;

        ws.internal_rhs.resize(block.n_internal);
        for (Dim i = 0; i < block.n_internal; ++i)
        {
            ws.internal_rhs[i] = rhs_local[internal_start + i];
        }

        ws.y.resize(block.n_internal);
        thomas_solve(block.a_internal, block.b_internal, block.c_internal,
                     ws.internal_rhs, ws.y);

        cache_out.has_internal = true;
        cache_out.y0 = ws.y.front();
        cache_out.yLast = ws.y.back();
    }

    // ---- Phase 2: interface RHS contribution (local) ----
    if (num_if == 0)
        return;

    if (schur_data_.has_left_interface)
    {
        const int iface_idx = schur_data_.get_interface_index(0);
        if (iface_idx >= 0 && iface_idx < num_if)
        {
            Real contrib = Real(0); // DO NOT include f_s[left] (as your original code)
            if (cache_out.has_internal)
            {
                contrib -= block.c_left_interface * cache_out.y0;
            }
            iface_rhs_out[iface_idx] += contrib;
        }
    }

    if (schur_data_.has_right_interface)
    {
        const int iface_idx = schur_data_.get_interface_index(1);
        if (iface_idx >= 0 && iface_idx < num_if)
        {
            Real contrib = rhs_local[local_N - 1]; // include f_s[right] once
            if (cache_out.has_internal)
            {
                contrib -= block.a_right_interface * cache_out.yLast;
            }
            iface_rhs_out[iface_idx] += contrib;
        }
    }
}

// ---------------------------
// Phase 3: ONE MPI_Allreduce
// ---------------------------
inline void SchurComplementSolver::allreduce_contribs(
    const std::vector<Real> &local_flat,
    std::vector<Real> &global_flat) const
{
    const int num_if = schur_data_.get_num_interfaces();
    if (num_if == 0)
    {
        global_flat.clear();
        return;
    }

    comm_.allreduce_interface_rhs_batched(local_flat, global_flat);
}

// ---------------------------
// Phase 4-5: NO MPI
// ---------------------------
inline void SchurComplementSolver::finalize_line(
    const Real *rhs_local, int local_N,
    const std::vector<Real> &global_iface_rhs, // size = num_interfaces
    Real *sol_local_out,                       // size = local_N
    ThreadWorkspace &ws)
{
    if (!schur_data_.is_preprocessed)
    {
        throw std::runtime_error("finalize_line: must preprocess() first");
    }
    if (local_N != schur_data_.local_N)
    {
        throw std::runtime_error("finalize_line: local_N mismatch");
    }

    const SchurBlockData &block = schur_data_.block_data;
    const int num_if = schur_data_.get_num_interfaces();

    // init output
    for (int i = 0; i < local_N; ++i)
        sol_local_out[i] = Real(0);

    // ---- Solve reduced interface system S u_s = f_s ----
    ws.iface_sol.assign(num_if, Real(0));
    if (num_if > 0)
    {
        solve_interface_system(global_iface_rhs, ws.iface_sol); // your function
    }

    // ---- Back-sub internal ----
    if (block.n_internal > 0)
    {
        const Dim internal_start = schur_data_.has_left_interface ? 1 : 0;

        ws.g.resize(block.n_internal);
        for (Dim i = 0; i < block.n_internal; ++i)
        {
            ws.g[i] = rhs_local[internal_start + i];
        }

        if (schur_data_.has_left_interface)
        {
            const int left_idx = schur_data_.get_interface_index(0);
            if (left_idx >= 0)
            {
                ws.g[0] -= block.a_internal[0] * ws.iface_sol[left_idx];
            }
        }

        if (schur_data_.has_right_interface)
        {
            const int right_idx = schur_data_.get_interface_index(1);
            if (right_idx >= 0)
            {
                ws.g[block.n_internal - 1] -=
                    block.c_internal[block.n_internal - 1] * ws.iface_sol[right_idx];
            }
        }

        ws.u_i.resize(block.n_internal);
        thomas_solve(block.a_internal, block.b_internal, block.c_internal,
                     ws.g, ws.u_i);

        for (Dim i = 0; i < block.n_internal; ++i)
        {
            sol_local_out[internal_start + i] = ws.u_i[i];
        }
    }

    // ---- Set interface unknowns into solution ----
    if (schur_data_.has_left_interface)
    {
        const int left_idx = schur_data_.get_interface_index(0);
        if (left_idx >= 0)
            sol_local_out[0] = ws.iface_sol[left_idx];
    }
    if (schur_data_.has_right_interface)
    {
        const int right_idx = schur_data_.get_interface_index(1);
        if (right_idx >= 0)
            sol_local_out[local_N - 1] = ws.iface_sol[right_idx];
    }

    // Single-proc case: solve full system
    if (num_if == 0)
    {
        // You can call your original full thomas on full (a_,b_,c_)
        std::vector<Real> rhs_vec(rhs_local, rhs_local + local_N);
        std::vector<Real> sol_vec;
        thomas_solve(a_, b_, c_, rhs_vec, sol_vec);
        for (int i = 0; i < local_N; ++i)
            sol_local_out[i] = sol_vec[i];
    }
}

// ---------------------------
// Full batched solve
// rhs_lines_flat: nLines * local_N
// sol_lines_flat: nLines * local_N
// ---------------------------
inline void SchurComplementSolver::solve_batch(
    const Real *rhs_lines_flat, int nLines,
    Real *sol_lines_flat,
    bool use_omp)
{
    if (!schur_data_.is_preprocessed)
    {
        throw std::runtime_error("solve_batch: must preprocess() first");
    }

    const int local_N = schur_data_.local_N;
    const int num_if = schur_data_.get_num_interfaces();

    std::vector<Real> iface_local(nLines * num_if, Real(0));
    std::vector<Real> iface_global;

    // optional cache (only needs y0,yLast); could be removed if you prefer
    std::vector<LineCache> cache(nLines);

// ============ Phase 1-2: parallel ============
#pragma omp parallel if (use_omp)
    {
        ThreadWorkspace ws;
        std::vector<Real> iface_rhs_line;

#pragma omp for schedule(static)
        for (int lid = 0; lid < nLines; ++lid)
        {
            const Real *rhs_line = rhs_lines_flat + lid * local_N;

            compute_local_contrib(rhs_line, local_N, iface_rhs_line, cache[lid], ws);

            // flatten
            for (int q = 0; q < num_if; ++q)
            {
                iface_local[lid * num_if + q] = iface_rhs_line[q];
            }
        }
    }

    // ============ Phase 3: one Allreduce ============
    if (num_if > 0)
    {
        allreduce_contribs(iface_local, iface_global);
    }

// ============ Phase 4-5: parallel ============
#pragma omp parallel if (use_omp)
    {
        ThreadWorkspace ws;
        std::vector<Real> iface_rhs_line;

#pragma omp for schedule(static)
        for (int lid = 0; lid < nLines; ++lid)
        {
            const Real *rhs_line = rhs_lines_flat + lid * local_N;
            Real *sol_line = sol_lines_flat + lid * local_N;

            if (num_if > 0)
            {
                iface_rhs_line.resize(num_if);
                for (int q = 0; q < num_if; ++q)
                {
                    iface_rhs_line[q] = iface_global[lid * num_if + q];
                }
            }
            else
            {
                iface_rhs_line.clear();
            }

            finalize_line(rhs_line, local_N, iface_rhs_line, sol_line, ws);
        }
    }
}

#endif // SCHUR_COMPLEMENT_SOLVER_HPP
