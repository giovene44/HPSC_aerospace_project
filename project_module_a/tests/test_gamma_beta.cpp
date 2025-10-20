#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include "Matrix3D.h"
#include "gamma_beta.h"

// ANSI color codes
#define COLOR_GREEN "\033[32m"
#define COLOR_RED "\033[31m"
#define COLOR_RESET "\033[0m"
#define COLOR_CYAN "\033[36m"
#define COLOR_YELLOW "\033[33m"

// Tolerance for floating point comparison
const float TOLERANCE = 1e-6f;

// Helper function to compare floats
bool float_equals(float a, float b, float tol = TOLERANCE) {
    return std::fabs(a - b) < tol;
}

// Test result printer
void print_test_result(const std::string& test_name, bool passed) {
    if (passed) {
        std::cout << COLOR_GREEN << "[PASS] " << COLOR_RESET << test_name << "\n";
    } else {
        std::cout << COLOR_RED << "[FAIL] " << COLOR_RESET << test_name << "\n";
    }
}

void test_gamma_basic() {
    std::cout << COLOR_CYAN << "\n=== Test: Gamma Basic Functionality ===" << COLOR_RESET << "\n";
    
    const size_t dim_x = 3, dim_y = 3, dim_z = 3;
    Matrix3D<float> k(dim_x, dim_y, dim_z);
    
    // Initialize k with known values
    for (size_t i = 0; i < dim_x; ++i) {
        for (size_t j = 0; j < dim_y; ++j) {
            for (size_t k_idx = 0; k_idx < dim_z; ++k_idx) {
                k(i, j, k_idx) = 5.0f;
            }
        }
    }
    
    float delta_t = 0.01f;
    float nu = 2.0f;
    
    Gamma gamma(dim_x, dim_y, dim_z);
    gamma.initialize(k, delta_t, nu);
    
    // Test point (0,0,0)
    float dt_nu_over_2 = (delta_t * nu) / 2.0f;
    float expected_beta = 1.0f + dt_nu_over_2 / k(0, 0, 0);
    float expected_gamma = dt_nu_over_2 / expected_beta;
    
    bool test1 = float_equals(gamma(0, 0, 0), expected_gamma);
    print_test_result("Gamma(0,0,0) = " + std::to_string(expected_gamma), test1);
    assert(test1);
    
    // Test point (1,1,1)
    expected_beta = 1.0f + dt_nu_over_2 / k(1, 1, 1);
    expected_gamma = dt_nu_over_2 / expected_beta;
    
    bool test2 = float_equals(gamma(1, 1, 1), expected_gamma);
    print_test_result("Gamma(1,1,1) = " + std::to_string(expected_gamma), test2);
    assert(test2);
    
    // Test point (2,2,2)
    expected_beta = 1.0f + dt_nu_over_2 / k(2, 2, 2);
    expected_gamma = dt_nu_over_2 / expected_beta;
    
    bool test3 = float_equals(gamma(2, 2, 2), expected_gamma);
    print_test_result("Gamma(2,2,2) = " + std::to_string(expected_gamma), test3);
    assert(test3);
}

void test_beta_basic() {
    std::cout << COLOR_CYAN << "\n=== Test: Beta Basic Functionality ===" << COLOR_RESET << "\n";
    
    const size_t dim_x = 3, dim_y = 3, dim_z = 3;
    Matrix3D<float> k(dim_x, dim_y, dim_z);
    
    // Initialize k with varying values
    float k_value = 2.0f;
    for (size_t i = 0; i < dim_x; ++i) {
        for (size_t j = 0; j < dim_y; ++j) {
            for (size_t k_idx = 0; k_idx < dim_z; ++k_idx) {
                k(i, j, k_idx) = k_value;
                k_value += 1.0f;
            }
        }
    }
    
    float delta_t = 0.02f;
    float nu = 1.5f;
    
    Beta beta(dim_x, dim_y, dim_z);
    beta.initialize(k, delta_t, nu);
    
    // Test point (0,0,0)
    float dt_nu_over_2 = (delta_t * nu) / 2.0f;
    float expected_beta = 1.0f + dt_nu_over_2 / k(0, 0, 0);
    
    bool test1 = float_equals(beta(0, 0, 0), expected_beta);
    print_test_result("Beta(0,0,0) = " + std::to_string(expected_beta), test1);
    assert(test1);
    
    // Test point (1,0,0)
    expected_beta = 1.0f + dt_nu_over_2 / k(1, 0, 0);
    
    bool test2 = float_equals(beta(1, 0, 0), expected_beta);
    print_test_result("Beta(1,0,0) = " + std::to_string(expected_beta), test2);
    assert(test2);
    
    // Test point (2,2,2)
    expected_beta = 1.0f + dt_nu_over_2 / k(2, 2, 2);
    
    bool test3 = float_equals(beta(2, 2, 2), expected_beta);
    print_test_result("Beta(2,2,2) = " + std::to_string(expected_beta), test3);
    assert(test3);
}

void test_beta_on_the_fly() {
    std::cout << COLOR_CYAN << "\n=== Test: Beta On-the-Fly Computation ===" << COLOR_RESET << "\n";
    
    const size_t dim_x = 2, dim_y = 2, dim_z = 2;
    Matrix3D<float> k(dim_x, dim_y, dim_z);
    
    // Initialize k
    for (size_t i = 0; i < dim_x; ++i) {
        for (size_t j = 0; j < dim_y; ++j) {
            for (size_t k_idx = 0; k_idx < dim_z; ++k_idx) {
                k(i, j, k_idx) = 5.0f;
            }
        }
    }
    
    float delta_t = 0.02f;
    float nu = 1.5f;
    
    Beta beta(dim_x, dim_y, dim_z);
    beta.initialize(k, delta_t, nu);
    
    // Initial value
    float dt_nu_over_2 = (delta_t * nu) / 2.0f;
    float expected_beta_initial = 1.0f + dt_nu_over_2 / 5.0f;
    
    bool test1 = float_equals(beta(0, 0, 0), expected_beta_initial);
    print_test_result("Beta(0,0,0) with k=5.0 = " + std::to_string(expected_beta_initial), test1);
    assert(test1);
    
    // Change k value
    k(0, 0, 0) = 10.0f;
    float expected_beta_new = 1.0f + dt_nu_over_2 / 10.0f;
    
    bool test2 = float_equals(beta(0, 0, 0), expected_beta_new);
    print_test_result("Beta(0,0,0) with k=10.0 = " + std::to_string(expected_beta_new), test2);
    assert(test2);
}

void test_gamma_stored_values() {
    std::cout << COLOR_CYAN << "\n=== Test: Gamma Stores Pre-computed Values ===" << COLOR_RESET << "\n";
    
    const size_t dim_x = 2, dim_y = 2, dim_z = 2;
    Matrix3D<float> k(dim_x, dim_y, dim_z);
    
    // Initialize k
    for (size_t i = 0; i < dim_x; ++i) {
        for (size_t j = 0; j < dim_y; ++j) {
            for (size_t k_idx = 0; k_idx < dim_z; ++k_idx) {
                k(i, j, k_idx) = 5.0f;
            }
        }
    }
    
    float delta_t = 0.02f;
    float nu = 1.5f;
    
    Gamma gamma(dim_x, dim_y, dim_z);
    gamma.initialize(k, delta_t, nu);
    
    // Store initial gamma value
    float stored_gamma = gamma(0, 0, 0);
    
    // Change k value
    k(0, 0, 0) = 10.0f;
    
    // Gamma should remain unchanged
    bool test1 = (gamma(0, 0, 0) == stored_gamma);
    print_test_result("Gamma(0,0,0) unchanged after k modification = " + std::to_string(stored_gamma), test1);
    assert(test1);
}

void test_edge_cases() {
    std::cout << COLOR_CYAN << "\n=== Test: Edge Cases ===" << COLOR_RESET << "\n";
    
    const size_t dim_x = 4, dim_y = 4, dim_z = 4;
    Matrix3D<float> k(dim_x, dim_y, dim_z);
    
    // Test with very small k values
    k(0, 0, 0) = 0.001f;
    
    // Test with large k values
    k(1, 1, 1) = 1000.0f;
    
    // Normal values
    for (size_t i = 2; i < dim_x; ++i) {
        for (size_t j = 0; j < dim_y; ++j) {
            for (size_t k_idx = 0; k_idx < dim_z; ++k_idx) {
                k(i, j, k_idx) = 5.0f;
            }
        }
    }
    
    float delta_t = 0.01f;
    float nu = 2.0f;
    
    Gamma gamma(dim_x, dim_y, dim_z);
    gamma.initialize(k, delta_t, nu);
    
    Beta beta(dim_x, dim_y, dim_z);
    beta.initialize(k, delta_t, nu);
    
    // Test small k
    float dt_nu_over_2 = (delta_t * nu) / 2.0f;
    float expected_beta = 1.0f + dt_nu_over_2 / 0.001f;
    float expected_gamma = dt_nu_over_2 / expected_beta;
    
    bool test1 = float_equals(beta(0, 0, 0), expected_beta);
    print_test_result("Beta(0,0,0) with small k=0.001", test1);
    assert(test1);
    
    bool test2 = float_equals(gamma(0, 0, 0), expected_gamma);
    print_test_result("Gamma(0,0,0) with small k=0.001", test2);
    assert(test2);
    
    // Test large k
    expected_beta = 1.0f + dt_nu_over_2 / 1000.0f;
    expected_gamma = dt_nu_over_2 / expected_beta;
    
    bool test3 = float_equals(beta(1, 1, 1), expected_beta);
    print_test_result("Beta(1,1,1) with large k=1000.0", test3);
    assert(test3);
    
    bool test4 = float_equals(gamma(1, 1, 1), expected_gamma);
    print_test_result("Gamma(1,1,1) with large k=1000.0", test4);
    assert(test4);
}

int main() {
    std::cout << std::fixed << std::setprecision(6);
    
    std::cout << COLOR_YELLOW << "\n╔════════════════════════════════════════╗\n";
    std::cout << "║  Gamma & Beta Test Suite              ║\n";
    std::cout << "╚════════════════════════════════════════╝" << COLOR_RESET << "\n";
    
    try {
        test_gamma_basic();
        test_beta_basic();
        test_beta_on_the_fly();
        test_gamma_stored_values();
        test_edge_cases();
        
        std::cout << COLOR_GREEN << "\n╔════════════════════════════════════════╗\n";
        std::cout << "║  ALL TESTS PASSED ✓                   ║\n";
        std::cout << "╚════════════════════════════════════════╝" << COLOR_RESET << "\n\n";
        
        return 0;
    } catch (const std::exception& e) {
        std::cout << COLOR_RED << "\n╔════════════════════════════════════════╗\n";
        std::cout << "║  TEST FAILED ✗                        ║\n";
        std::cout << "╚════════════════════════════════════════╝" << COLOR_RESET << "\n";
        std::cout << "Error: " << e.what() << "\n\n";
        return 1;
    }
}