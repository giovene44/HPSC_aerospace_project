#include <vector>
#include <iostream>

void thomas_algorithm(const std::vector<float> &a, const std::vector<float> &b, const std::vector<float> &c, const std::vector<float> &rhs, std::vector<float> &x)
{
    int n = rhs.size();
    std::vector<float> c_prime(c.size(), 0.0);
    std::vector<float> rhs_prime(n, 0.0);

    c_prime[0] = c[0] / b[0];
    rhs_prime[0] = rhs[0] / b[0];

    for (int i = 1; i < n - 1; ++i)
    {
        float m = 1.0 / (b[i] - a[i - 1] * c_prime[i - 1]);
        c_prime[i] = c[i] * m;
        rhs_prime[i] = (rhs[i] - a[i - 1] * rhs_prime[i - 1]) * m;
    }

    float m = 1.0 / (b[n - 1] - a[n - 2] * c_prime[n - 2]);
    rhs_prime[n - 1] = (rhs[n - 1] - a[n - 2] * rhs_prime[n - 2]) * m;

    x[n - 1] = rhs_prime[n - 1];

    for (int i = n - 2; i >= 0; --i)
    {
        x[i] = rhs_prime[i] - c_prime[i] * x[i + 1];
    }
}
