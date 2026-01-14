#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include "Variables.hpp"

/**
 * @brief Reference Thomas algorithm (serial tridiagonal solver)
 */
void thomas_solve(const std::vector<Real>& a,
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

int main(int argc, char** argv) {
    // Default problem size and iteration count
    int N = 100;
    int num_iterations = 100;

    if (argc > 1) {
        N = std::atoi(argv[1]);
    }
    if (argc > 2) {
        num_iterations = std::atoi(argv[2]);
    }

    // Setup tridiagonal system: -u_{i-1} + 2*u_i - u_{i+1} = sin(i*pi/N)
    std::vector<Real> a(N, -1.0);
    std::vector<Real> b(N, 2.0);
    std::vector<Real> c(N, -1.0);
    std::vector<Real> rhs(N);

    // Boundary conditions
    a[0] = 0.0;
    c[N-1] = 0.0;

    // RHS: For -u'' = sin(πx), we need h² * f on the right-hand side
    const Real pi = 3.14159265358979323846;
    Real h = 1.0 / (N - 1);
    Real h2 = h * h;
    for (int i = 0; i < N; ++i) {
        Real x = i * h;
        rhs[i] = h2 * std::sin(pi * x);
    }

    // Time multiple solve iterations (to match parallel benchmark)
    std::vector<Real> x;
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < num_iterations; ++iter) {
        thomas_solve(a, b, c, rhs, x);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    // Print average timing per solve in seconds (6 decimal places)
    double avg_time = elapsed.count() / num_iterations;
    std::cout << std::fixed << std::setprecision(6) << avg_time << std::endl;

    return 0;
}
