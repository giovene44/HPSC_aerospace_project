#ifndef GAMMA_BETA_H_
#define GAMMA_BETA_H_

#include "Matrix3D.h"
#include <memory>

/**
 * Gamma class
 * Pre-computes and stores γ = (Δt·ν)/(2·β) for all grid points
 * where β = 1 + (Δt·ν)/(2·k)
 * 
 * Memory layout: Uses Matrix3D internally with row-major order
 * Data is computed once during initialization and stored for fast access
 */
class Gamma {
 private:
    Matrix3D<float> gamma_data_;  // Stores pre-computed gamma values
    size_t dim_x_;
    size_t dim_y_;
    size_t dim_z_;

 public:
    /**
     * Constructor: Initializes gamma matrix with given dimensions
     * @param dim_x: dimension along x-axis
     * @param dim_y: dimension along y-axis
     * @param dim_z: dimension along z-axis
     */
    Gamma(size_t dim_x, size_t dim_y, size_t dim_z);

    /**
     * Initialize gamma values: γ = (Δt·ν)/(2·β)
     * where β = 1 + (Δt·ν)/(2·k)
     * 
     * @param k: 3D matrix of conductivity values
     * @param delta_t: time step
     * @param nu: diffusion coefficient
     */
    void initialize(const Matrix3D<float>& k, float delta_t, float nu);

    /**
     * Access gamma value at position (i, j, k)
     * Returns pre-computed value from stored matrix
     */
    float operator()(size_t i, size_t j, size_t k) const;

    // Get dimensions
    size_t dim_x() const { return dim_x_; }
    size_t dim_y() const { return dim_y_; }
    size_t dim_z() const { return dim_z_; }
};


/**
 * Beta class
 * Computes β = 1 + (Δt·ν)/(2·k) on-the-fly when accessed
 * 
 * Memory layout: Stores reference to k matrix and parameters
 * Does NOT store computed beta values - recalculates each time
 */
class Beta {
 private:
    const Matrix3D<float>* k_ptr_;  // Pointer to k matrix (not owned)
    float delta_t_;
    float nu_;
    float dt_nu_over_2_;  // Pre-computed (Δt·ν)/2 for efficiency
    size_t dim_x_;
    size_t dim_y_;
    size_t dim_z_;

 public:
    /**
     * Constructor: Initializes beta computation parameters
     * @param dim_x: dimension along x-axis
     * @param dim_y: dimension along y-axis
     * @param dim_z: dimension along z-axis
     */
    Beta(size_t dim_x, size_t dim_y, size_t dim_z);

    /**
     * Initialize beta parameters
     * Stores references and parameters needed for on-the-fly computation
     * 
     * @param k: 3D matrix of conductivity values (must remain valid)
     * @param delta_t: time step
     * @param nu: diffusion coefficient
     */
    void initialize(const Matrix3D<float>& k, float delta_t, float nu);

    /**
     * Access beta value at position (i, j, k)
     * Computes β = 1 + (Δt·ν)/(2·k(i,j,k)) on-the-fly
     * 
     * @param i: index along x-axis
     * @param j: index along y-axis
     * @param k: index along z-axis
     * @return: computed beta value
     */
    float operator()(size_t i, size_t j, size_t k) const;

    // Get dimensions
    size_t dim_x() const { return dim_x_; }
    size_t dim_y() const { return dim_y_; }
    size_t dim_z() const { return dim_z_; }
};


// ============================================================================
// Implementation
// ============================================================================

// Gamma implementation
Gamma::Gamma(size_t dim_x, size_t dim_y, size_t dim_z)
    : gamma_data_(dim_x, dim_y, dim_z),
      dim_x_(dim_x),
      dim_y_(dim_y),
      dim_z_(dim_z) {}

void Gamma::initialize(const Matrix3D<float>& k, float delta_t, float nu) {
    float dt_nu_over_2 = (delta_t * nu) / 2.0f;
    
    // Pre-compute all gamma values and store them
    for (size_t i = 0; i < dim_x_; ++i) {
        for (size_t j = 0; j < dim_y_; ++j) {
            for (size_t k_idx = 0; k_idx < dim_z_; ++k_idx) {
                float k_val = k(i, j, k_idx);
                float beta = 1.0f + dt_nu_over_2 / k_val;
                float gamma = dt_nu_over_2 / beta;
                gamma_data_(i, j, k_idx) = gamma;
            }
        }
    }
}

float Gamma::operator()(size_t i, size_t j, size_t k) const {
    return gamma_data_(i, j, k);
}


// Beta implementation
Beta::Beta(size_t dim_x, size_t dim_y, size_t dim_z)
    : k_ptr_(nullptr),
      delta_t_(0.0f),
      nu_(0.0f),
      dt_nu_over_2_(0.0f),
      dim_x_(dim_x),
      dim_y_(dim_y),
      dim_z_(dim_z) {}

void Beta::initialize(const Matrix3D<float>& k, float delta_t, float nu) {
    k_ptr_ = &k;
    delta_t_ = delta_t;
    nu_ = nu;
    dt_nu_over_2_ = (delta_t * nu) / 2.0f;  // Pre-compute for efficiency
}

float Beta::operator()(size_t i, size_t j, size_t k) const {
    // Compute beta on-the-fly: β = 1 + (Δt·ν)/(2·k)
    float k_val = (*k_ptr_)(i, j, k);
    return 1.0f + dt_nu_over_2_ / k_val;
}

#endif  // GAMMA_BETA_H_