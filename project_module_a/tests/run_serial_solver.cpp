#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
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
    // Default problem size
    int N = 100;

    if (argc > 1) {
        N = std::atoi(argv[1]);
    }

    std::cout << "Serial Thomas Solver" << std::endl;
    std::cout << "Problem size: N = " << N << std::endl;

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

    // Solve
    std::vector<Real> x;
    thomas_solve(a, b, c, rhs, x);

    // Save solution to file
    std::string filename = "solution_serial_N" + std::to_string(N) + ".dat";
    std::ofstream out(filename);
    out.precision(16);

    for (int i = 0; i < N; ++i) {
        out << i << " " << x[i] << "\n";
    }
    out.close();

    std::cout << "Solution saved to: " << filename << std::endl;
    std::cout << "First 5 values: ";
    for (int i = 0; i < std::min(5, N); ++i) {
        std::cout << x[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
