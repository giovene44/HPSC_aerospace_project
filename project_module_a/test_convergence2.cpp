#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using Real = float;

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
    
    c_prime[0] = c[0] / b[0];
    d_prime[0] = rhs[0] / b[0];
    
    for (size_t i = 1; i < n; ++i) {
        Real denom = b[i] - a[i] * c_prime[i - 1];
        c_prime[i] = c[i] / denom;
        d_prime[i] = (rhs[i] - a[i] * d_prime[i - 1]) / denom;
    }
    
    x[n - 1] = d_prime[n - 1];
    for (size_t i = n - 1; i > 0; --i) {
        x[i - 1] = d_prime[i - 1] - c_prime[i - 1] * x[i];
    }
}

// For the problem: -u'' = sin(πx), u(0) = u(1) = 0
// Exact solution: u(x) = sin(πx) / π²
Real exact_solution(Real x) {
    const Real pi = 3.14159265358979323846;
    return std::sin(pi * x) / (pi * pi);
}

int main() {
    const Real pi = 3.14159265358979323846;
    
    std::cout << "Testing with CORRECTED RHS (including h²)\n\n";
    
    std::vector<int> sizes = {10, 20, 40, 80, 160};
    std::vector<Real> errors;
    
    for (int N : sizes) {
        Real h = 1.0 / (N - 1);
        Real h2 = h * h;
        
        std::vector<Real> a(N, -1.0);
        std::vector<Real> b(N, 2.0);
        std::vector<Real> c(N, -1.0);
        std::vector<Real> rhs(N);
        
        a[0] = 0.0;
        c[N-1] = 0.0;
        
        for (int i = 0; i < N; ++i) {
            Real x_coord = i * h;
            rhs[i] = h2 * std::sin(pi * x_coord);  // CORRECTED with h²
        }
        
        std::vector<Real> x;
        thomas_solve(a, b, c, rhs, x);
        
        // Compute error at interior points
        Real max_error = 0.0;
        for (int i = 1; i < N-1; ++i) {
            Real x_coord = i * h;
            Real exact = exact_solution(x_coord);
            Real error = std::abs(x[i] - exact);
            max_error = std::max(max_error, error);
        }
        
        errors.push_back(max_error);
        std::cout << "N=" << N << ", h=" << h << ", max_error=" << max_error;
        
        if (errors.size() > 1) {
            Real ratio = errors[errors.size()-2] / errors[errors.size()-1];
            std::cout << ", reduction_ratio=" << ratio;
        }
        std::cout << std::endl;
    }
    
    return 0;
}
