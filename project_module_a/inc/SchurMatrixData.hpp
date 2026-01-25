#ifndef SCHUR_MATRIX_DATA_HPP
#define SCHUR_MATRIX_DATA_HPP

#include "Variables.hpp"
#include <vector>
#include <array>

/**
 * @brief Stores precomputed Schur complement data for a single local block.
 *
 * For a block with n_internal internal points and 2 interface points (left, right),
 * this stores the contributions needed to assemble the Schur complement matrix S.
 *
 * The Schur complement is: S = A_ss - A_si * A_ii^(-1) * A_is
 * where:
 *   - A_ii: tridiagonal matrix for internal points
 *   - A_ss: diagonal matrix for interface points
 *   - A_is, A_si: sparse coupling matrices (2 non-zeros per block)
 */
struct SchurBlockData {
    // Size of internal block (n_internal = local_N - 2 for blocks with 2 interfaces)
    Dim n_internal;

    // Columns of A_ii^(-1) * A_is
    // These are computed by solving A_ii * x = e_i for the interface-connected columns
    // inv_Aii_col_left[i] = (A_ii^(-1))_{i,0} * a[1] (contribution from left interface)
    // inv_Aii_col_right[i] = (A_ii^(-1))_{i,n-1} * c[n] (contribution from right interface)
    std::vector<Real> inv_Aii_col_left;
    std::vector<Real> inv_Aii_col_right;

    // Original tridiagonal coefficients for internal block
    // Used for back-substitution phase
    std::vector<Real> a_internal;  // Lower diagonal (size: n_internal)
    std::vector<Real> b_internal;  // Main diagonal (size: n_internal)
    std::vector<Real> c_internal;  // Upper diagonal (size: n_internal)

    // Interface coupling coefficients from original matrix
    Real a_left_interface;   // a[0]: coupling left_interface -> first_internal
    Real c_left_interface;   // c[0]: coupling left_interface -> right neighbor's interface
    Real a_right_interface;  // a[N-1]: coupling right_interface -> last_internal
    Real c_right_interface;  // c[N-1]: coupling right_interface -> right neighbor's interface

    // Original diagonal values at interfaces
    Real b_left_interface;   // b[0]
    Real b_right_interface;  // b[N-1]

    // Schur complement diagonal contributions (local part)
    // schur_diag_left = b[0] - c[0] * (A_ii^(-1) * A_is)[0,left]
    // schur_diag_right = b[N-1] - a[N-1] * (A_ii^(-1) * A_is)[n-1,right]
    Real schur_diag_left;
    Real schur_diag_right;

    // Schur complement off-diagonal contributions (coupling to neighbors)
    // These couple this block's interfaces to neighbor block's interfaces
    Real schur_offdiag_to_left;   // -a[0]: couples to left neighbor's right interface
    Real schur_offdiag_to_right;  // -c[N-1]: couples to right neighbor's left interface

    SchurBlockData() : n_internal(0), a_left_interface(0), c_left_interface(0),
                       a_right_interface(0), c_right_interface(0),
                       b_left_interface(0), b_right_interface(0),
                       schur_diag_left(0), schur_diag_right(0),
                       schur_offdiag_to_left(0), schur_offdiag_to_right(0) {}
};

/**
 * @brief Container for all Schur complement data for parallel tridiagonal solve.
 *
 * Stores process topology, local domain info, and the assembled reduced system.
 */
struct SchurComplementData {
    // Process topology
    int num_procs;      // Total number of processes in solve direction
    int my_rank;        // This process's rank (0 to num_procs-1)

    // Local domain info
    Dim local_N;        // Local size including interface overlap points
    Dim global_N;       // Global problem size
    Dim global_start;   // Starting global index for this process's internal points

    // Number of internal points (excluding interfaces)
    Dim n_internal;     // = local_N - 2 for interior processes
                        // = local_N - 1 for boundary processes

    // Interface indices (local coordinates)
    // For process 0: left_interface = -1 (no left interface), right_interface = local_N - 1
    // For process N-1: left_interface = 0, right_interface = -1 (no right interface)
    // For interior: left_interface = 0, right_interface = local_N - 1
    int left_interface_local;
    int right_interface_local;

    // Whether this process has left/right interfaces (boundary processes don't)
    bool has_left_interface;
    bool has_right_interface;

    // Precomputed block data for local solve
    SchurBlockData block_data;

    // Assembled reduced system (Schur complement matrix S)
    // Size: num_interfaces = num_procs - 1 (interior interfaces only)
    // The reduced system is tridiagonal:
    //   schur_a[i] * u_{i-1} + schur_b[i] * u_i + schur_c[i] * u_{i+1} = schur_rhs[i]
    std::vector<Real> schur_a;  // Lower diagonal of reduced system
    std::vector<Real> schur_b;  // Main diagonal of reduced system
    std::vector<Real> schur_c;  // Upper diagonal of reduced system

    // Workspace for reduced system RHS (assembled from all processes)
    std::vector<Real> schur_rhs;

    // Solution at interfaces (after solving reduced system)
    std::vector<Real> interface_solution;

    // Flag indicating if preprocessing is complete
    bool is_preprocessed;

    SchurComplementData() : num_procs(1), my_rank(0), local_N(0), global_N(0),
                            global_start(0), n_internal(0),
                            left_interface_local(-1), right_interface_local(-1),
                            has_left_interface(false), has_right_interface(false),
                            is_preprocessed(false) {}

    /**
     * @brief Initialize topology and allocate arrays.
     * @param global_size Global problem size
     * @param nprocs Number of processes
     * @param rank This process's rank
     */
    void init(Dim global_size, int nprocs, int rank) {
        global_N = global_size;
        num_procs = nprocs;
        my_rank = rank;

        if (nprocs == 1) {
            // Single process - no decomposition needed
            local_N = global_N;
            global_start = 0;
            n_internal = global_N;
            has_left_interface = false;
            has_right_interface = false;
            left_interface_local = -1;
            right_interface_local = -1;
        } else {
            // Multi-process with proper interface sharing
            // Partition the domain so adjacent processes share interface points
            Dim points_per_proc = global_N / nprocs;
            Dim remainder = global_N % nprocs;

            // Compute non-overlapping partition endpoints
            Dim my_start = rank * points_per_proc + std::min(rank, static_cast<int>(remainder));
            Dim my_end = (rank + 1) * points_per_proc + std::min(rank + 1, static_cast<int>(remainder)) - 1;

            // global_start is the starting index for local storage
            // For rank 0, it's 0; for others, it's one before my_start (to include left interface)
            if (rank == 0) {
                global_start = 0;
                local_N = my_end - my_start + 1;  // Own [my_start..my_end]
            } else {
                global_start = my_start - 1;  // Include left interface point
                local_N = my_end - global_start + 1;  // Own [my_start-1..my_end]
            }

            // Set interface flags and indices
            if (rank == 0) {
                has_left_interface = false;
                has_right_interface = true;
                left_interface_local = -1;
                right_interface_local = local_N - 1;
                n_internal = local_N - 1;
            } else if (rank == num_procs - 1) {
                has_left_interface = true;
                has_right_interface = false;
                left_interface_local = 0;
                right_interface_local = -1;
                n_internal = local_N - 1;
            } else {
                has_left_interface = true;
                has_right_interface = true;
                left_interface_local = 0;
                right_interface_local = local_N - 1;
                n_internal = local_N - 2;
            }
        }

        // Allocate reduced system arrays (size = num_procs - 1 interfaces)
        int num_interfaces = num_procs - 1;
        if (num_interfaces > 0) {
            schur_a.resize(num_interfaces, 0.0);
            schur_b.resize(num_interfaces, 0.0);
            schur_c.resize(num_interfaces, 0.0);
            schur_rhs.resize(num_interfaces, 0.0);
            interface_solution.resize(num_interfaces, 0.0);
        }

        is_preprocessed = false;
    }

    /**
     * @brief Get the number of interface unknowns in the reduced system.
     */
    int get_num_interfaces() const {
        return num_procs - 1;
    }

    /**
     * @brief Get this process's interface index in the reduced system.
     * @param side 0 for left interface, 1 for right interface
     * @return Interface index (0 to num_interfaces-1), or -1 if no such interface
     */
    int get_interface_index(int side) const {
        if (side == 0 && has_left_interface) {
            return my_rank - 1;  // Left interface is shared with process my_rank-1
        } else if (side == 1 && has_right_interface) {
            return my_rank;      // Right interface is at global interface index my_rank
        }
        return -1;
    }
};

#endif // SCHUR_MATRIX_DATA_HPP
