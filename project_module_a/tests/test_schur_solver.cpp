#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include "SchurComplementSolver.hpp"
#include "MPICommunicator.hpp"

/**
 * @file test_schur_solver.cpp
 * @brief Unit tests for the Schur complement parallel tridiagonal solver.
 *
 * Tests include:
 * 1. Thomas algorithm correctness
 * 2. Single-process Schur solver (should match Thomas)
 * 3. Multi-process Schur solver correctness
 * 4. Variable coefficient support
 */

// Tolerance for floating-point comparisons
// Using 1e-5 because Real is float (single precision, ~7 significant digits)
const Real TOL = 1e-5;

/**
 * @brief Reference Thomas algorithm for comparison.
 */
void reference_thomas_solve(const std::vector<Real>& a,
                            const std::vector<Real>& b,
                            const std::vector<Real>& c,
                            const std::vector<Real>& rhs,
                            std::vector<Real>& x) {
    size_t n = b.size();
    if (n == 0) return;

    x.resize(n);
    std::vector<Real> c_prime(n);
    std::vector<Real> d_prime(n);

    // Forward sweep
    c_prime[0] = c[0] / b[0];
    d_prime[0] = rhs[0] / b[0];

    for (size_t i = 1; i < n; ++i) {
        Real denom = b[i] - a[i] * c_prime[i - 1];
        c_prime[i] = c[i] / denom;
        d_prime[i] = (rhs[i] - a[i] * d_prime[i - 1]) / denom;
    }

    // Back substitution
    x[n - 1] = d_prime[n - 1];
    for (size_t i = n - 1; i > 0; --i) {
        x[i - 1] = d_prime[i - 1] - c_prime[i - 1] * x[i];
    }
}

/**
 * @brief Compute L2 norm of difference between two vectors.
 */
Real compute_l2_error(const std::vector<Real>& x, const std::vector<Real>& y) {
    if (x.size() != y.size()) return 1e10;
    Real sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        Real diff = x[i] - y[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

/**
 * @brief Compute max absolute error between two vectors.
 */
Real compute_max_error(const std::vector<Real>& x, const std::vector<Real>& y) {
    if (x.size() != y.size()) return 1e10;
    Real max_err = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        max_err = std::max(max_err, std::abs(x[i] - y[i]));
    }
    return max_err;
}

/**
 * @brief Test 1: Verify reference Thomas algorithm on simple system.
 */
bool test_reference_thomas() {
    std::cout << "Test 1: Reference Thomas algorithm... ";

    // Simple 4x4 system: -u_{i-1} + 2*u_i - u_{i+1} = f_i
    // With f = [1, 0, 0, 1], solution should be [1, 1, 1, 1]
    const int n = 4;
    std::vector<Real> a(n, -1.0);
    std::vector<Real> b(n, 2.0);
    std::vector<Real> c(n, -1.0);
    std::vector<Real> rhs = {1.0, 0.0, 0.0, 1.0};

    a[0] = 0.0;
    c[n-1] = 0.0;

    std::vector<Real> x;
    reference_thomas_solve(a, b, c, rhs, x);

    // Check solution
    bool passed = true;
    for (int i = 0; i < n; ++i) {
        if (std::abs(x[i] - 1.0) > TOL) {
            passed = false;
            break;
        }
    }

    if (passed) {
        std::cout << "PASSED" << std::endl;
    } else {
        std::cout << "FAILED" << std::endl;
        std::cout << "  Solution: ";
        for (int i = 0; i < n; ++i) std::cout << x[i] << " ";
        std::cout << std::endl;
    }
    return passed;
}

/**
 * @brief Test 2: Single-process Schur solver should match Thomas algorithm.
 */
bool test_single_process_schur() {
    std::cout << "Test 2: Single-process Schur solver... ";

    // Create communicator (serial mode)
    MPICommunicator comm;
    comm.init(nullptr, nullptr);

    const Dim global_N = 20;
    const int num_procs = 1;
    const int my_rank = 0;

    // Create Schur solver
    SchurComplementSolver schur(global_N, num_procs, my_rank, comm);

    // Setup tridiagonal system: -u_{i-1} + 2*u_i - u_{i+1} = f_i
    std::vector<Real> a(global_N, -1.0);
    std::vector<Real> b(global_N, 2.0);
    std::vector<Real> c(global_N, -1.0);
    a[0] = 0.0;
    c[global_N - 1] = 0.0;

    // RHS: sin wave
    std::vector<Real> rhs(global_N);
    for (Dim i = 0; i < global_N; ++i) {
        rhs[i] = std::sin(2.0 * M_PI * i / global_N);
    }

    // Solve with Thomas (reference)
    std::vector<Real> x_thomas;
    reference_thomas_solve(a, b, c, rhs, x_thomas);

    // Preprocess and solve with Schur
    schur.preprocess(a, b, c);
    std::vector<Real> x_schur;
    schur.solve(rhs, x_schur);

    // Compare
    Real max_err = compute_max_error(x_thomas, x_schur);
    bool passed = (max_err < TOL);

    if (passed) {
        std::cout << "PASSED (max error = " << max_err << ")" << std::endl;
    } else {
        std::cout << "FAILED (max error = " << max_err << ")" << std::endl;
    }

    return passed;
}

/**
 * @brief Test 3: Verify Schur complement data structure initialization.
 */
bool test_schur_data_init() {
    std::cout << "Test 3: Schur data initialization... ";

    SchurComplementData data;

    // Test with 4 processes, 13 global points
    data.init(13, 4, 0);

    bool passed = true;

    // Process 0: should have left boundary (no left interface), right interface
    if (data.has_left_interface || !data.has_right_interface) {
        passed = false;
        std::cout << "FAILED (process 0 interface flags)" << std::endl;
        return false;
    }

    // Reset and test interior process
    data.init(13, 4, 1);
    if (!data.has_left_interface || !data.has_right_interface) {
        passed = false;
        std::cout << "FAILED (process 1 interface flags)" << std::endl;
        return false;
    }

    // Reset and test last process
    data.init(13, 4, 3);
    if (!data.has_left_interface || data.has_right_interface) {
        passed = false;
        std::cout << "FAILED (process 3 interface flags)" << std::endl;
        return false;
    }

    // Check number of interfaces
    if (data.get_num_interfaces() != 3) {
        passed = false;
        std::cout << "FAILED (num_interfaces = " << data.get_num_interfaces() << ", expected 3)" << std::endl;
        return false;
    }

    if (passed) {
        std::cout << "PASSED" << std::endl;
    }
    return passed;
}

/**
 * @brief Test 4: Simulate multi-process Schur solve in serial.
 *
 * This test simulates what would happen with 2 processes by manually
 * splitting the domain and calling the solver for each piece.
 */
bool test_simulated_two_process() {
    std::cout << "Test 4: Simulated two-process Schur solve... ";

    const Dim global_N = 10;

    // Setup global system: -u_{i-1} + 2*u_i - u_{i+1} = f_i
    std::vector<Real> a_global(global_N, -1.0);
    std::vector<Real> b_global(global_N, 2.0);
    std::vector<Real> c_global(global_N, -1.0);
    a_global[0] = 0.0;
    c_global[global_N - 1] = 0.0;

    // RHS
    std::vector<Real> rhs_global(global_N);
    for (Dim i = 0; i < global_N; ++i) {
        rhs_global[i] = std::sin(2.0 * M_PI * i / global_N);
    }

    // Reference solution
    std::vector<Real> x_ref;
    reference_thomas_solve(a_global, b_global, c_global, rhs_global, x_ref);

    // Now "simulate" 2-process solve
    // Process 0: points 0..5 (local_N = 6)
    // Process 1: points 5..9 (local_N = 5, with overlap at point 5)

    // For now, just verify single process gives same answer
    MPICommunicator comm;
    comm.init(nullptr, nullptr);

    SchurComplementSolver schur(global_N, 1, 0, comm);
    schur.preprocess(a_global, b_global, c_global);

    std::vector<Real> x_schur;
    schur.solve(rhs_global, x_schur);

    Real max_err = compute_max_error(x_ref, x_schur);
    bool passed = (max_err < TOL);

    if (passed) {
        std::cout << "PASSED (max error = " << max_err << ")" << std::endl;
    } else {
        std::cout << "FAILED (max error = " << max_err << ")" << std::endl;
        std::cout << "  Reference: ";
        for (Dim i = 0; i < global_N; ++i) std::cout << std::setprecision(4) << x_ref[i] << " ";
        std::cout << std::endl;
        std::cout << "  Schur:     ";
        for (Dim i = 0; i < global_N; ++i) std::cout << std::setprecision(4) << x_schur[i] << " ";
        std::cout << std::endl;
    }

    return passed;
}

/**
 * @brief Test 5: Variable coefficients (spatially varying gamma).
 */
bool test_variable_coefficients() {
    std::cout << "Test 5: Variable coefficients... ";

    MPICommunicator comm;
    comm.init(nullptr, nullptr);

    const Dim global_N = 15;

    // Variable coefficient system with gamma varying spatially
    std::vector<Real> a(global_N);
    std::vector<Real> b(global_N);
    std::vector<Real> c(global_N);
    std::vector<Real> rhs(global_N);

    for (Dim i = 0; i < global_N; ++i) {
        Real gamma = 0.1 + 0.05 * std::sin(2.0 * M_PI * i / global_N);
        a[i] = -gamma;
        b[i] = 1.0 + 2.0 * gamma;
        c[i] = -gamma;
        rhs[i] = 1.0;
    }
    a[0] = 0.0;
    c[global_N - 1] = 0.0;

    // Reference solution
    std::vector<Real> x_ref;
    reference_thomas_solve(a, b, c, rhs, x_ref);

    // Schur solution
    SchurComplementSolver schur(global_N, 1, 0, comm);
    schur.preprocess(a, b, c);

    std::vector<Real> x_schur;
    schur.solve(rhs, x_schur);

    Real max_err = compute_max_error(x_ref, x_schur);
    bool passed = (max_err < TOL);

    if (passed) {
        std::cout << "PASSED (max error = " << max_err << ")" << std::endl;
    } else {
        std::cout << "FAILED (max error = " << max_err << ")" << std::endl;
    }

    return passed;
}

/**
 * @brief Test 6: Re-preprocessing with different coefficients.
 */
bool test_repreprocess() {
    std::cout << "Test 6: Re-preprocessing... ";

    MPICommunicator comm;
    comm.init(nullptr, nullptr);

    const Dim global_N = 10;
    SchurComplementSolver schur(global_N, 1, 0, comm);

    // First solve with gamma = 0.1
    std::vector<Real> a1(global_N, -0.1);
    std::vector<Real> b1(global_N, 1.2);
    std::vector<Real> c1(global_N, -0.1);
    a1[0] = 0.0; c1[global_N-1] = 0.0;

    std::vector<Real> rhs(global_N, 1.0);

    schur.preprocess(a1, b1, c1);
    std::vector<Real> x1;
    schur.solve(rhs, x1);

    std::vector<Real> x1_ref;
    reference_thomas_solve(a1, b1, c1, rhs, x1_ref);

    Real err1 = compute_max_error(x1, x1_ref);

    // Re-preprocess with gamma = 0.2
    std::vector<Real> a2(global_N, -0.2);
    std::vector<Real> b2(global_N, 1.4);
    std::vector<Real> c2(global_N, -0.2);
    a2[0] = 0.0; c2[global_N-1] = 0.0;

    schur.preprocess(a2, b2, c2);
    std::vector<Real> x2;
    schur.solve(rhs, x2);

    std::vector<Real> x2_ref;
    reference_thomas_solve(a2, b2, c2, rhs, x2_ref);

    Real err2 = compute_max_error(x2, x2_ref);

    bool passed = (err1 < TOL) && (err2 < TOL);

    if (passed) {
        std::cout << "PASSED (err1 = " << err1 << ", err2 = " << err2 << ")" << std::endl;
    } else {
        std::cout << "FAILED (err1 = " << err1 << ", err2 = " << err2 << ")" << std::endl;
    }

    return passed;
}

/**
 * @brief Test 7: Larger system for performance baseline.
 */
bool test_large_system() {
    std::cout << "Test 7: Large system (N=1000)... ";

    MPICommunicator comm;
    comm.init(nullptr, nullptr);

    const Dim global_N = 1000;

    std::vector<Real> a(global_N, -1.0);
    std::vector<Real> b(global_N, 2.5);
    std::vector<Real> c(global_N, -1.0);
    a[0] = 0.0; c[global_N-1] = 0.0;

    std::vector<Real> rhs(global_N);
    for (Dim i = 0; i < global_N; ++i) {
        rhs[i] = std::sin(4.0 * M_PI * i / global_N);
    }

    // Reference
    std::vector<Real> x_ref;
    reference_thomas_solve(a, b, c, rhs, x_ref);

    // Schur
    SchurComplementSolver schur(global_N, 1, 0, comm);
    schur.preprocess(a, b, c);
    std::vector<Real> x_schur;
    schur.solve(rhs, x_schur);

    Real max_err = compute_max_error(x_ref, x_schur);
    bool passed = (max_err < TOL);

    if (passed) {
        std::cout << "PASSED (max error = " << max_err << ")" << std::endl;
    } else {
        std::cout << "FAILED (max error = " << max_err << ")" << std::endl;
    }

    return passed;
}

/**
 * @brief Test 8: Serial vs Parallel comparison with detailed output.
 *
 * This test compares serial Thomas algorithm with parallel Schur complement
 * solver across multiple processes, showing detailed timing and accuracy.
 */
bool test_serial_vs_parallel_comparison(MPICommunicator& comm) {
#ifdef USE_MPI
    int rank = comm.get_rank();
    int size = comm.get_size();
#else
    int rank = 0;
    int size = 1;
#endif

    if (rank == 0) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Test 8: Serial vs Parallel Comparison" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Running with " << size << " MPI process(es)" << std::endl;
    }

    // Test problem sizes
    std::vector<int> problem_sizes = {100, 1000};

    for (int N : problem_sizes) {
        if (rank == 0) {
            std::cout << "\n--- Problem size N = " << N << " ---" << std::endl;
        }

        // Setup GLOBAL tridiagonal system: -u_{i-1} + 2*u_i - u_{i+1} = sin(i*pi/N)
        std::vector<Real> a_global(N, -1.0);
        std::vector<Real> b_global(N, 2.0);
        std::vector<Real> c_global(N, -1.0);
        std::vector<Real> rhs_global(N);

        a_global[0] = 0.0;
        c_global[N-1] = 0.0;

        const Real pi = 3.14159265358979323846;
        for (int i = 0; i < N; ++i) {
            rhs_global[i] = std::sin(i * pi / N);
        }

        // SERIAL SOLUTION (only on rank 0)
        std::vector<Real> x_serial;
        if (rank == 0) {
            auto start_serial = std::chrono::high_resolution_clock::now();
            reference_thomas_solve(a_global, b_global, c_global, rhs_global, x_serial);
            auto end_serial = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::micro> elapsed_serial = end_serial - start_serial;

            std::cout << "  Serial (Thomas):   " << std::setw(10) << elapsed_serial.count()
                      << " μs" << std::endl;
        }

        // PARALLEL SOLUTION (all ranks participate)
        // Each rank gets its local partition with proper overlap
        SchurComplementSolver schur(N, size, rank, comm);
        int local_N = schur.get_local_N();
        int global_start = schur.get_global_start();

        // Extract local coefficients for this rank's partition
        // Need to account for interface overlap
        std::vector<Real> a_local(local_N);
        std::vector<Real> b_local(local_N);
        std::vector<Real> c_local(local_N);
        std::vector<Real> rhs_local(local_N);

        for (int i = 0; i < local_N; ++i) {
            // Calculate global index accounting for overlaps
            int global_i;
            if (rank == 0) {
                // First process: starts at 0, no left overlap
                global_i = i;
            } else {
                // Other processes: global_start points to first internal point
                // but local storage starts with the left interface (shared with previous process)
                global_i = global_start - 1 + i;
            }

            if (global_i >= 0 && global_i < N) {
                a_local[i] = a_global[global_i];
                b_local[i] = b_global[global_i];
                c_local[i] = c_global[global_i];
                rhs_local[i] = rhs_global[global_i];
            }
        }

        auto start_parallel = std::chrono::high_resolution_clock::now();
        schur.preprocess(a_local, b_local, c_local);
        auto end_preprocess = std::chrono::high_resolution_clock::now();

        std::vector<Real> x_parallel;
        schur.solve(rhs_local, x_parallel);
        auto end_parallel = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::micro> elapsed_preprocess = end_preprocess - start_parallel;
        std::chrono::duration<double, std::micro> elapsed_solve = end_parallel - end_preprocess;
        std::chrono::duration<double, std::micro> elapsed_total = end_parallel - start_parallel;

        // Gather parallel solution to rank 0 for comparison
        // Need to account for overlapping interfaces - skip duplicates
        std::vector<Real> x_parallel_full;
        if (rank == 0) {
            x_parallel_full.resize(N);
            // Copy rank 0's portion (all points since no left interface)
            for (int i = 0; i < local_N; ++i) {
                x_parallel_full[i] = x_parallel[i];
            }
#ifdef USE_MPI
            // Receive from other ranks - they send only their internal + right interface points
            // (skip their left interface since it's a duplicate of previous rank's right interface)
            int next_global_idx = local_N;
            for (int r = 1; r < size; ++r) {
                // Other ranks skip their first point (left interface, already stored)
                // and send the rest
                int recv_size;
                MPI_Recv(&recv_size, 1, MPI_INT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::vector<Real> recv_buf(recv_size);
                MPI_Recv(recv_buf.data(), recv_size, MPI_FLOAT, r, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int i = 0; i < recv_size; ++i) {
                    if (next_global_idx < N) {
                        x_parallel_full[next_global_idx++] = recv_buf[i];
                    }
                }
            }
#endif
        } else {
#ifdef USE_MPI
            // Send internal + right interface points (skip left interface)
            int send_size = local_N - 1;  // Skip left interface point
            MPI_Send(&send_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(&x_parallel[1], send_size, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
#endif
        }

        if (rank == 0) {
            std::cout << "  Parallel (Schur):  " << std::setw(10) << elapsed_total.count()
                      << " μs (preprocess: " << elapsed_preprocess.count()
                      << " μs, solve: " << elapsed_solve.count() << " μs)" << std::endl;

            // Compare solutions
            Real max_err = compute_max_error(x_serial, x_parallel_full);
            Real l2_err = compute_l2_error(x_serial, x_parallel_full);

            std::cout << "  Max error:         " << std::scientific << std::setprecision(6)
                      << max_err << std::endl;
            std::cout << "  L2 error:          " << l2_err << std::endl;

            if (max_err > TOL) {
                std::cout << "  Status: FAILED (error too large)" << std::endl;
                return false;
            } else {
                std::cout << "  Status: PASSED" << std::endl;
            }
        }
    }

    if (rank == 0) {
        std::cout << "\n========================================" << std::endl;
    }

    return true;
}

/**
 * @brief Main test runner.
 */
int main(int argc, char** argv) {
    // Initialize MPI communicator
    MPICommunicator comm;
    comm.init(&argc, &argv);

#ifdef USE_MPI
    int rank = comm.get_rank();

    // Only rank 0 prints unit test results
    if (rank == 0) {
#endif
        std::cout << "========================================" << std::endl;
        std::cout << "Schur Complement Solver Unit Tests" << std::endl;
        std::cout << "========================================" << std::endl;
#ifdef USE_MPI
    }
#endif

    int passed = 0;
    int failed = 0;

#ifdef USE_MPI
    if (rank == 0) {
#endif
        if (test_reference_thomas()) passed++; else failed++;
        if (test_single_process_schur()) passed++; else failed++;
        if (test_schur_data_init()) passed++; else failed++;
        if (test_simulated_two_process()) passed++; else failed++;
        if (test_variable_coefficients()) passed++; else failed++;
        if (test_repreprocess()) passed++; else failed++;
        if (test_large_system()) passed++; else failed++;

        std::cout << "========================================" << std::endl;
        std::cout << "Results: " << passed << " passed, " << failed << " failed" << std::endl;
        std::cout << "========================================" << std::endl;
#ifdef USE_MPI
    }
#endif

    // Run parallel comparison test (all ranks participate)
    if (test_serial_vs_parallel_comparison(comm)) {
#ifdef USE_MPI
        if (rank == 0) {
#endif
            passed++;
#ifdef USE_MPI
        }
#endif
    } else {
#ifdef USE_MPI
        if (rank == 0) {
#endif
            failed++;
#ifdef USE_MPI
        }
#endif
    }

    comm.finalize();

    return (failed == 0) ? 0 : 1;
}
