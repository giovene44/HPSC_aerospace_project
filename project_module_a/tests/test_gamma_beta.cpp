#include <iostream>
#include <iomanip>
#include <cmath>
#include "Matrix3D.h"
#include "gamma_beta.h"

// Test 1: Basic functionality test with small 3D matrix
void test1_basic_functionality() {
    std::cout << "========================================\n";
    std::cout << "TEST 1: Basic Functionality Test\n";
    std::cout << "========================================\n\n";

    // Create a small 3x3x3 matrix for k values
    const size_t dim_x = 3, dim_y = 3, dim_z = 3;
    Matrix3D<float> k(dim_x, dim_y, dim_z);

    // Initialize k with simple values (avoid zero!)
    float k_value = 1.0f;
    for (size_t i = 0; i < dim_x; ++i) {
        for (size_t j = 0; j < dim_y; ++j) {
            for (size_t k_idx = 0; k_idx < dim_z; ++k_idx) {
                k(i, j, k_idx) = k_value;
                k_value += 0.5f;
            }
        }
    }

    // Set parameters
    float delta_t = 0.01f;
    float nu = 2.0f;

    // Create and initialize Gamma
    Gamma gamma(dim_x, dim_y, dim_z);
    gamma.initialize(k, delta_t, nu);

    // Create and initialize Beta
    Beta beta(dim_x, dim_y, dim_z);
    beta.initialize(k, delta_t, nu);

    // Test a few positions
    std::cout << "Testing at position (0, 0, 0):\n";
    std::cout << "  k(0,0,0) = " << k(0, 0, 0) << "\n";
    std::cout << "  beta(0,0,0) = " << beta(0, 0, 0) << "\n";
    std::cout << "  gamma(0,0,0) = " << gamma(0, 0, 0) << "\n";

    // Manual calculation for verification
    float dt_nu_over_2 = (delta_t * nu) / 2.0f;
    float expected_beta = 1.0f + dt_nu_over_2 / k(0, 0, 0);
    float expected_gamma = dt_nu_over_2 / expected_beta;
    std::cout << "  Expected beta = " << expected_beta << "\n";
    std::cout << "  Expected gamma = " << expected_gamma << "\n\n";

    std::cout << "Testing at position (1, 1, 1):\n";
    std::cout << "  k(1,1,1) = " << k(1, 1, 1) << "\n";
    std::cout << "  beta(1,1,1) = " << beta(1, 1, 1) << "\n";
    std::cout << "  gamma(1,1,1) = " << gamma(1, 1, 1) << "\n";

    expected_beta = 1.0f + dt_nu_over_2 / k(1, 1, 1);
    expected_gamma = dt_nu_over_2 / expected_beta;
    std::cout << "  Expected beta = " << expected_beta << "\n";
    std::cout << "  Expected gamma = " << expected_gamma << "\n\n";

    std::cout << "Testing at position (2, 2, 2):\n";
    std::cout << "  k(2,2,2) = " << k(2, 2, 2) << "\n";
    std::cout << "  beta(2,2,2) = " << beta(2, 2, 2) << "\n";
    std::cout << "  gamma(2,2,2) = " << gamma(2, 2, 2) << "\n";

    expected_beta = 1.0f + dt_nu_over_2 / k(2, 2, 2);
    expected_gamma = dt_nu_over_2 / expected_beta;
    std::cout << "  Expected beta = " << expected_beta << "\n";
    std::cout << "  Expected gamma = " << expected_gamma << "\n\n";

    std::cout << "TEST 1: PASSED\n\n";
}


// Test 2: Verify that Beta computes on-the-fly (values change when k changes)
void test2_on_the_fly_computation() {
    std::cout << "========================================\n";
    std::cout << "TEST 2: On-the-Fly Computation Test\n";
    std::cout << "========================================\n\n";

    const size_t dim_x = 2, dim_y = 2, dim_z = 2;
    Matrix3D<float> k(dim_x, dim_y, dim_z);

    // Initialize k with uniform value
    for (size_t i = 0; i < dim_x; ++i) {
        for (size_t j = 0; j < dim_y; ++j) {
            for (size_t k_idx = 0; k_idx < dim_z; ++k_idx) {
                k(i, j, k_idx) = 5.0f;
            }
        }
    }

    float delta_t = 0.02f;
    float nu = 1.5f;

    // Initialize Beta
    Beta beta(dim_x, dim_y, dim_z);
    beta.initialize(k, delta_t, nu);

    std::cout << "Initial k(0,0,0) = " << k(0, 0, 0) << "\n";
    std::cout << "Initial beta(0,0,0) = " << beta(0, 0, 0) << "\n\n";

    // Change k value
    k(0, 0, 0) = 10.0f;
    std::cout << "After changing k(0,0,0) to " << k(0, 0, 0) << ":\n";
    std::cout << "beta(0,0,0) = " << beta(0, 0, 0) << "\n";
    std::cout << "(Beta should reflect the new k value!)\n\n";

    // Verify with manual calculation
    float dt_nu_over_2 = (delta_t * nu) / 2.0f;
    float expected_beta = 1.0f + dt_nu_over_2 / k(0, 0, 0);
    std::cout << "Expected beta = " << expected_beta << "\n";
    
    float tolerance = 1e-6f;
    bool test_passed = std::fabs(beta(0, 0, 0) - expected_beta) < tolerance;
    
    if (test_passed) {
        std::cout << "\nTEST 2: PASSED - Beta correctly recomputes with new k values\n\n";
    } else {
        std::cout << "\nTEST 2: FAILED\n\n";
    }

    // Test Gamma (should NOT change since it stores pre-computed values)
    std::cout << "========================================\n";
    std::cout << "Bonus: Gamma stores pre-computed values\n";
    std::cout << "========================================\n\n";
    
    // Reset k
    for (size_t i = 0; i < dim_x; ++i) {
        for (size_t j = 0; j < dim_y; ++j) {
            for (size_t k_idx = 0; k_idx < dim_z; ++k_idx) {
                k(i, j, k_idx) = 5.0f;
            }
        }
    }

    Gamma gamma(dim_x, dim_y, dim_z);
    gamma.initialize(k, delta_t, nu);
    
    std::cout << "Initial k(0,0,0) = " << k(0, 0, 0) << "\n";
    std::cout << "Initial gamma(0,0,0) = " << gamma(0, 0, 0) << "\n\n";
    
    float stored_gamma = gamma(0, 0, 0);
    
    // Change k value
    k(0, 0, 0) = 10.0f;
    std::cout << "After changing k(0,0,0) to " << k(0, 0, 0) << ":\n";
    std::cout << "gamma(0,0,0) = " << gamma(0, 0, 0) << "\n";
    std::cout << "(Gamma should remain unchanged!)\n\n";
    
    bool gamma_unchanged = (gamma(0, 0, 0) == stored_gamma);
    if (gamma_unchanged) {
        std::cout << "VERIFIED: Gamma correctly stores pre-computed values\n\n";
    } else {
        std::cout << "ERROR: Gamma value changed unexpectedly\n\n";
    }
}


int main() {
    std::cout << std::fixed << std::setprecision(6);
    
    test1_basic_functionality();
    test2_on_the_fly_computation();
    
    std::cout << "========================================\n";
    std::cout << "All tests completed!\n";
    std::cout << "========================================\n";
    
    return 0;
}


