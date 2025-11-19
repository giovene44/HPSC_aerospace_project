#include <string>
#include <cmath>
#include <functional>
#include <vector>
#include "ScalarVariable.hpp"
#include "BoundaryFunctions.hpp"
class ParseInput
{

public:
    ParseInput() {};

    void parse_input(const std::string &input_file)
    {
        std::ifstream file(input_file);
        if (!file.is_open())
        {
            std::cerr << "Error - Cannot open file " << input_file << std::endl;
            return;
        }
        std::string token;
        auto next_value = [&](auto &var)
        {
            while (file >> token)
            {
                if (token[0] == '#')
                {
                    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                    continue;
                }
                std::istringstream(token) >> var;
                return;
            }
        };

        // ========= Mesh dimensions ==========
        next_value(Nx);
        next_value(Ny);
        next_value(Nz);
        // std::cout << "Parsed Nx, Ny, Nz: " << Nx << ", " << Ny << ", " << Nz << std::endl;

        // ========== Time parameters ==========
        next_value(dt);
        next_value(T);
        // std::cout << "Parsed dt, T: " << dt << ", " << T << std::endl;

        // ========== Spatial parameters ==========
        next_value(dx);
        next_value(dy);
        next_value(dz);
        // std::cout << "Parsed dx, dy, dz: " << dx << ", " << dy << ", " << dz << std::endl;

        // ========= Initial values ==========
        next_value(u_boundary_file);
        next_value(p_boundary_file);
        // std::cout << "Parsed u_boundary_file, p_boundary_file: " << u_boundary_file << ", " << p_boundary_file << std::endl;
        //  next_value(k_file);
        // std::cout << "Parsed k_file: " << k_file << std::endl;

        // std::cout << "Boundary conditions parsing completed.\n";
        //  ========= OUTPUT ==========
        std::cout << "\n===== Input Parameters Loaded =====\n";
        std::cout << "Mesh points (Nx, Ny, Nz): " << Nx << ", " << Ny << ", " << Nz << std::endl;
        std::cout << "Time step size (dt):       " << dt << std::endl;
        std::cout << "Total simulation time (T): " << T << std::endl;
        std::cout << "Finite diff step (dx,dy,dz): "
                  << dx << ", " << dy << ", " << dz << std::endl;
        std::cout << "Initial u0 file:           " << u_boundary_file << std::endl;
        std::cout << "Initial p0 file:           " << p_boundary_file << std::endl;
        // std::cout << "k values file:             " << k_file << std::endl;
        std::cout << "===================================\n\n";
    }

    Real dt; // Time step

    Dim Nx; // Grid points in x
    Dim Ny; // Grid points in y
    Dim Nz; // Grid points in z

    Real dx; // Grid spacing in x
    Real dy; // Grid spacing in y
    Real dz; // Grid spacing in z

    Real T;

    std::string u_boundary_file; // Initial condition file for velocity
    std::string p_boundary_file; // Initial condition file for pressure
};