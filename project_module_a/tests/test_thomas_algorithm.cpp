#include <vector>
#include "Variables.hpp"
#include "ScalarVariable.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>

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
};
Real f(Real x)
{
    return 0.0f;
}

int main()
{
    int N = 5;
    Real dn = Real(M_PI / N);

    // (I-dxx) (sin(x)) = f(x) -> f(x) = sin(x) + sin''(x) = sin(x) - sin(x) = 0

    std::vector<Real> a(N, Real(-1.0f) / (dn * dn));
    std::vector<Real> b(N, Real(1.0f) + (Real(2.0f) / (dn * dn)));
    std::vector<Real> c(N, Real(-1.0f) / (dn * dn));
    std::vector<Real> d(N);
    std::vector<Real> x(N);
    std::vector<Real> expected_x(N);

    // Neumann BC x'(0) = alpha and x'(π) = beta
    Real alpha = 0.0; // example
    Real beta = 0.0;  // example

    // interior coefficients remain the same
    for (int i = 1; i < N - 1; ++i)
    {
        a[i] = -1.0 / (dn * dn);
        b[i] = 1.0 + 2.0 / (dn * dn);
        c[i] = -1.0 / (dn * dn);
        d[i] = f(i * dn); // or whatever your source term is
    }

    // --- left boundary (Neumann)
    a[0] = 0.0;
    b[0] = -2.0 / (dn * dn) - 1.0;
    c[0] = 2.0 / (dn * dn);
    d[0] = f(0) + 2.0 * alpha / dn;

    // --- right boundary (Neumann)
    a[N - 1] = 2.0 / (dn * dn);
    b[N - 1] = -2.0 / (dn * dn) - 1.0;
    c[N - 1] = 0.0;
    d[N - 1] = f(M_PI) - 2.0 * beta / dn;

    std::cout << "Matrix A (" << N << "x" << N << "):\n";
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            Real val = Real(0.0);
            if (j == i)
                val = b[i];
            else if (j == i - 1)
                val = a[i];
            else if (j == i + 1)
                val = c[i];
            std::cout << std::setw(12) << std::setprecision(6) << std::fixed << static_cast<double>(val);
        }
        std::cout << '\n';
    }

    std::cout << "RHS d:\n";
    for (int i = 0; i < N; ++i)
        std::cout << std::setw(12) << std::setprecision(6) << std::fixed << static_cast<double>(d[i]);
    std::cout << '\n';

    thomas_algorithm(a, b, c, d, x);

    Real tolerance = 1e-6;
    for (size_t i = 0; i < x.size(); ++i)
    {
        if (std::abs(x[i] - expected_x[i]) > tolerance)
        {
            std::cerr << "[ERROR] Mismatch at index " << i << ": expected " << expected_x[i] << ", got " << x[i] << std::endl;
            return -1;
        }
    }

    return 0;
}