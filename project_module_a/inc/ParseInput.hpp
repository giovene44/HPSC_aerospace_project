#include <string>
#include <cmath>
#include <functional>
#include <vector>
#include <fstream>   // Needed for std::ifstream
#include <iostream>  // Needed for std::cerr, std::cout
#include <sstream>   // Needed for std::istringstream
#include <limits>    // Needed for std::numeric_limits
#include <stdexcept> // Needed for std::runtime_error (recommended for error handling)

// Assuming Real and Dim are defined in ScalarVariable.hpp
// using Real = double;
// using Dim = int;

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
            // IMPORTANT: Throwing an exception is safer than returning silently.
            // This prevents the program from continuing with uninitialized data.
            throw std::runtime_error("Error - Cannot open input file: " + input_file);
        }

        std::string token;
        auto next_value = [&](auto &var)
        {
            while (file >> token)
            {
                if (token.empty() || token[0] == '#') // Check for empty tokens or comments
                {
                    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                    continue;
                }
                std::istringstream(token) >> var;
                return;
            }
            // Optional: throw an error if file ends unexpectedly while reading a value
        };

        // ========= Mesh dimensions and other parameters ==========
        next_value(Nx);
        next_value(Ny);
        next_value(Nz);
        next_value(dt);
        next_value(T);
        next_value(dx);
        next_value(dy);
        next_value(dz);
        next_value(u_boundary_file);
        next_value(p_boundary_file);

        // ========= OUTPUT ==========
        std::cout << "\n===== Input Parameters Loaded =====\n";
        std::cout << "Mesh points (Nx, Ny, Nz): " << Nx << ", " << Ny << ", " << Nz << std::endl;
        std::cout << "Time step size (dt):       " << dt << std::endl;
        std::cout << "Total simulation time (T): " << T << std::endl;
        std::cout << "Finite diff step (dx,dy,dz): "
                  << dx << ", " << dy << ", " << dz << std::endl;
        std::cout << "Initial u0 file:           " << u_boundary_file << std::endl;
        std::cout << "Initial p0 file:           " << p_boundary_file << std::endl;
        std::cout << "===================================\n\n";
    }

    // Public member variables (no change)
    Real dt;
    Dim Nx;
    Dim Ny;
    Dim Nz;
    Real dx;
    Real dy;
    Real dz;
    Real T;
    std::string u_boundary_file;
    std::string p_boundary_file;
};