#include <string>
#include <cmath>
#include <functional>
#include <vector>
#include <fstream>   // Needed for std::ifstream
#include <iostream>  // Needed for std::cerr, std::cout
#include <sstream>   // Needed for std::istringstream
#include <limits>    // Needed for std::numeric_limits
#include <stdexcept> // Needed for std::runtime_error

// Assuming Real and Dim are defined (e.g., using Real = double; using Dim = int;)

class ParseInput
{
private:
    // 1. Private Constructor: Prevents direct creation of instances
    ParseInput() = default;

    // 2. Delete Copy/Move Operations: Prevents cloning the singleton
    ParseInput(const ParseInput &) = delete;
    ParseInput &operator=(const ParseInput &) = delete;
    ParseInput(ParseInput &&) = delete;
    ParseInput &operator=(ParseInput &&) = delete;

public:
    // 3. Public Static Method: Provides global access to the single instance
    static ParseInput &getInstance()
    {
        // Guaranteed to be thread-safe in C++11 and later
        static ParseInput instance;
        return instance;
    }

    void parse_input(const std::string &input_file)
    {
        std::ifstream file(input_file);
        if (!file.is_open())
        {
            // Throw an exception on failure
            throw std::runtime_error("Error - Cannot open input file: " + input_file);
        }

        std::string token;
        auto next_value = [&](auto &var)
        {
            while (file >> token)
            {
                if (token.empty() || token[0] == '#') // Skip comments
                {
                    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                    continue;
                }
                std::istringstream(token) >> var;
                return;
            }
            // If we reach here, we hit EOF before reading the expected value
            throw std::runtime_error("Error - Unexpected end of file while reading input parameters.");
        };

        // ===========================================
        // 1. Read Domain Dimensions (DimX, DimY, DimZ)
        // ===========================================
        next_value(DimX);
        next_value(DimY);
        next_value(DimZ);

        // ===========================================
        // 2. Read Mesh Points (Nx, Ny, Nz)
        // ===========================================
        next_value(Nx);
        next_value(Ny);
        next_value(Nz);

        // ===========================================
        // 3. Read Time Parameters (dt, T)
        // ===========================================
        next_value(dt);
        next_value(T);

        // ===========================================
        // 4. Calculate dx, dy, dz (Grid Spacing)
        // ===========================================
        // Grid spacing is calculated as Domain Dimension / (Number of points - 1)
        // This is standard for non-periodic domains where the first and last points
        // are included in the grid (e.g., 0 and DimX).
        if (Nx > 1)
        {
            dx = DimX / (Real)(Nx - 1);
        }
        else
        {
            dx = DimX;
        }
        if (Ny > 1)
        {
            dy = DimY / (Real)(Ny - 1);
        }
        else
        {
            dy = DimY;
        }
        if (Nz > 1)
        {
            dz = DimZ / (Real)(Nz - 1);
        }
        else
        {
            dz = DimZ;
        }

        // ===========================================
        // 5. Read Boundary Files
        // ===========================================
        next_value(u_boundary_file);
        next_value(p_boundary_file);

        // ========= OUTPUT ==========
        std::cout << "\n===== Input Parameters Loaded =====\n";
        std::cout << "Domain Dimensions (X, Y, Z): " << DimX << ", " << DimY << ", " << DimZ << std::endl;
        std::cout << "Mesh points (Nx, Ny, Nz): " << Nx << ", " << Ny << ", " << Nz << std::endl;
        std::cout << "Time step size (dt):       " << dt << std::endl;
        std::cout << "Total simulation time (T): " << T << std::endl;
        std::cout << "Calculated step (dx,dy,dz): "
                  << dx << ", " << dy << ", " << dz << std::endl;
        std::cout << "Initial u0 file:           " << u_boundary_file << std::endl;
        std::cout << "Initial p0 file:           " << p_boundary_file << std::endl;
        std::cout << "===================================\n\n";
    }

    // Public member variables
    Real DimX; // Domain dimension in X
    Real DimY; // Domain dimension in Y
    Real DimZ; // Domain dimension in Z

    Dim Nx; // Grid points in X
    Dim Ny; // Grid points in Y
    Dim Nz; // Grid points in Z

    Real dt; // Time step
    Real T;  // Total simulation time

    // Calculated grid spacing (now derived, not read)
    Real dx;
    Real dy;
    Real dz;

    std::string u_boundary_file; // Initial condition file for velocity
    std::string p_boundary_file; // Initial condition file for pressure
};