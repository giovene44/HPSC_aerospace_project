#include <cmath>
#include <vector>

class ManufacturedSolution {
public:
    // Constructor with default Reynolds number = 100
    ManufacturedSolution(float reynolds_number = 100.0f) : Re(reynolds_number) {}
    
    // Set Reynolds number
    void setReynoldsNumber(float reynolds_number) {
        Re = reynolds_number;
    }
    
    // Get Reynolds number
    float getReynoldsNumber() const {
        return Re;
    }
    
    // Compute velocity field [u, v, w]
    std::vector<float> velocity(float x, float y, float z, float t) const {
        std::vector<float> u(3);
        
        float sin_x = std::sin(x);
        float cos_x = std::cos(x);
        float sin_y = std::sin(y);
        float cos_y = std::cos(y);
        float sin_z = std::sin(z);
        float cos_z = std::cos(z);
        float sin_t = std::sin(t);
        
        u[0] = sin_t * sin_x * sin_y * sin_z;
        u[1] = sin_t * cos_x * cos_y * cos_z;
        u[2] = sin_t * cos_x * sin_y * (sin_z + cos_z);
        
        return u;
    }
    
    // Compute pressure
    float pressure(float x, float y, float z) const {
        float cos_x = std::cos(x);
        float sin_y = std::sin(y);
        float sin_z = std::sin(z);
        float cos_z = std::cos(z);
        
        return (-3.0f / Re) * cos_x * sin_y * (sin_z - cos_z);
    }
    
    // Compute coefficient k
    float coefficient(float x, float y, float z) const {
        return std::sin(x) * std::sin(y) * std::sin(z);
    }
    
    // Compute forcing term [fx, fy, fz]
    std::vector<float> forcing(float x, float y, float z, float t) const {
        std::vector<float> result(3);
        
        // Precompute common trigonometric values
        float sin_x = std::sin(x);
        float cos_x = std::cos(x);
        float sin_y = std::sin(y);
        float cos_y = std::cos(y);
        float sin_z = std::sin(z);
        float cos_z = std::cos(z);
        float sin_t = std::sin(t);
        float cos_t = std::cos(t);
        
        // Compute k
        float k_val = coefficient(x, y, z);
        
        // Compute u, v, w values
        float u_val = sin_t * sin_x * sin_y * sin_z;
        float v_val = sin_t * cos_x * cos_y * cos_z;
        float w_val = sin_t * cos_x * sin_y * (sin_z + cos_z);
        
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
    float Re;  // Reynolds number
};