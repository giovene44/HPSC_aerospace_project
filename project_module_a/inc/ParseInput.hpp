#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include <stdexcept>
#include <sstream>
#include <limits>
#include <type_traits> // For std::is_same_v
#include "mpParser.h"  // Required for expression parsing

// --- ParseInput Class (Singleton) ---
// --- ParseInput Class (Singleton) ---
class ParseInput
{
private:
    ParseInput() = default;

public:
    static ParseInput &getInstance()
    {
        static ParseInput instance;
        return instance;
    }
    ParseInput(const ParseInput &) = delete;
    ParseInput &operator=(const ParseInput &) = delete;

    // --- Member Variables ---
    int num_runs;

    Real DimX, DimY, DimZ;
    Dim Nx, Ny, Nz;
    Real dt, T;
    Real dx, dy, dz;
    std::string u_boundary_file, p_boundary_file;

    void parse_input(const std::string &input_file)
    {
        std::ifstream file(input_file);
        if (!file.is_open())
        {
            throw std::runtime_error("Error - Cannot open input file: " + input_file);
        }

        // Initialize Math Parser
        mup::ParserX p;

        // Safely define constants. Some might already exist.
        try
        {
            p.DefineConst("pi", M_PI);
        }
        catch (...)
        {
        }
        try
        {
            p.DefineConst("Pi", M_PI);
        }
        catch (...)
        {
        }
        try
        {
            p.DefineConst("PI", M_PI);
        }
        catch (...)
        {
        }

        std::string token;

        // Helper to skip comments (#) and read next value
        // Now supports mathematical expressions for numeric types
        auto next_value = [&](auto &var)
        {
            using T = std::decay_t<decltype(var)>;

            while (file >> token)
            {
                if (token.empty() || token[0] == '#')
                {
                    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                    continue;
                }

                // If the target variable is a string, just copy the token
                if constexpr (std::is_same_v<T, std::string>)
                {
                    var = token;
                }
                // If the target is numeric (Real or Dim), evaluate the expression
                else
                {
                    try
                    {
                        p.SetExpr(token);

                        // Evaluate the expression
                        mup::Value ans = p.Eval();

                        // FIX: Removed .IsFloat()/.IsInteger() checks which caused the compilation error.
                        // Directly get the value as float and cast to the target type T (int or double).
                        // This works for both 10 (int) and 2*pi (float).
                        var = static_cast<T>(ans.GetFloat());
                    }
                    catch (const mup::ParserError &e)
                    {
                        std::cerr << "Parser Error Message: " << e.GetMsg() << std::endl;
                        throw std::runtime_error("Input Parse Error for token '" + token + "'");
                    }
                    catch (const std::exception &e)
                    {
                        throw std::runtime_error("Standard Exception in Parse: " + std::string(e.what()));
                    }
                }
                return;
            }
            throw std::runtime_error("Error - Unexpected end of file while reading input parameters.");
        };

        // ===========================================
        // 0. Read Number of Runs
        // ===========================================
        next_value(num_runs);

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

        std::cout << "\n===== Input Parameters Loaded =====\n";
        std::cout << "Convergence Runs:          " << num_runs << std::endl;
        std::cout << "Domain Dimensions:         " << DimX << ", " << DimY << ", " << DimZ << std::endl;
        std::cout << "Base Mesh (Nx, Ny, Nz):    " << Nx << ", " << Ny << ", " << Nz << std::endl;
        std::cout << "Base dt:                   " << dt << std::endl;
        std::cout << "===================================\n";
    }
};

// ---------------------------------------------