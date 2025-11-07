#include <cmath>
#include <vector>
#include "Variables.hpp"

class ManufacturedSolution {
public:
    // Constructor with default Reynolds number = 100
    ManufacturedSolution(Real reynolds_number = 100.0f) : Re(reynolds_number) {}
    
    // Set Reynolds number
    void setReynoldsNumber(Real reynolds_number) {
        Re = reynolds_number;
    }
    
    // Get Reynolds number
    Real getReynoldsNumber() const {
        return Re;
    }
    
    // Compute velocity field [u, v, w]
    std::vector<Real> velocity(Real x, Real y, Real z, Real t) const {
        std::vector<Real> u(3);
        
        Real sin_x = std::sin(x);
        Real cos_x = std::cos(x);
        Real sin_y = std::sin(y);
        Real cos_y = std::cos(y);
        Real sin_z = std::sin(z);
        Real cos_z = std::cos(z);
        Real sin_t = std::sin(t);
        
        u[0] = sin_t * sin_x * sin_y * sin_z;
        u[1] = sin_t * cos_x * cos_y * cos_z;
        u[2] = sin_t * cos_x * sin_y * (sin_z + cos_z);
        
        return u;
    }
    
    // Compute pressure
    Real pressure(Real x, Real y, Real z) const {
        Real cos_x = std::cos(x);
        Real sin_y = std::sin(y);
        Real sin_z = std::sin(z);
        Real cos_z = std::cos(z);
        
        return (-3.0f / Re) * cos_x * sin_y * (sin_z - cos_z);
    }
    
    // Compute coefficient k
    Real coefficient(Real x, Real y, Real z) const {
        return std::sin(x) * std::sin(y) * std::sin(z);
    }
    
    // Compute forcing term [fx, fy, fz]
    std::vector<Real> forcing(Real x, Real y, Real z, Real t) const {
        std::vector<Real> result(3);
        
        // Precompute common trigonometric values
        Real sin_x = std::sin(x);
        Real cos_x = std::cos(x);
        Real sin_y = std::sin(y);
        Real cos_y = std::cos(y);
        Real sin_z = std::sin(z);
        Real cos_z = std::cos(z);
        Real sin_t = std::sin(t);
        Real cos_t = std::cos(t);
        
        // Compute k
        Real k_val = coefficient(x, y, z);
        
        // Compute u, v, w values
        Real u_val = sin_t * sin_x * sin_y * sin_z;
        Real v_val = sin_t * cos_x * cos_y * cos_z;
        Real w_val = sin_t * cos_x * sin_y * (sin_z + cos_z);
        
        // f_x component
        result[0] = cos_t * sin_x * sin_y * sin_z
                    + (3.0f / Re) * sin_t * sin_x * sin_y * sin_z
                    + k_val * u_val
                    + (3.0f / Re) * sin_x * sin_y * (sin_z - cos_z);
        
        // f_y component
        result[1] = cos_t * cos_x * cos_y * cos_z
                    + (3.0f / Re) * sin_t * cos_x * cos_y * cos_z
                    + k_val * v_val
                    - (3.0f / Re) * cos_x * cos_y * (sin_z - cos_z);
        
        // f_z component
        result[2] = cos_t * cos_x * sin_y * (sin_z + cos_z)
                    + (3.0f / Re) * sin_t * cos_x * sin_y * (sin_z + cos_z)
                    + k_val * w_val
                    - (3.0f / Re) * cos_x * sin_y * (cos_z + sin_z);
        
        return result;
    }
    
private:
    Real Re;  // Reynolds number
};