#ifndef PARSE_INPUT_HPP
#define PARSE_INPUT_HPP

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include <stdexcept>
#include <sstream>
#include <limits>
#include <functional>
#include <type_traits>
#include "mpParser.h"
#include "Variables.hpp"

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

    std::string u_boundary_file;
    std::string p_boundary_file;

    // RAW STRINGS for Forcing Functions
    std::string fx_expression;
    std::string fy_expression;
    std::string fz_expression;

    // RAW STRINGS for Exact Solution (u, v, w, p)
    std::string u_exact_expression;
    std::string v_exact_expression;
    std::string w_exact_expression;
    std::string p_exact_expression;

    // PHYSICS PARAMETERS (Read from input)
    std::string k_expression; // Can be "1e10" or "sin(x)"
    Real nu;                  // Explicit viscosity (e.g., 6.0)

    // MPI DECOMPOSITION PARAMETERS (Optional, -1 for automatic)
    int Px, Py, Pz;

    // --- Helpers to generate std::function objects ---
    std::function<std::vector<Real>(Real, Real, Real, Real)> get_forcing_function() const;
    std::function<std::vector<Real>(Real, Real, Real, Real)> get_exact_velocity_function() const;
    std::function<Real(Real, Real, Real, Real)> get_exact_pressure_function() const;

    // Helper for the K function
    std::function<Real(Real, Real, Real)> get_k_function() const;

    void parse_input(const std::string &input_file)
    {
        std::ifstream file(input_file);
        if (!file.is_open())
        {
            throw std::runtime_error("Error - Cannot open input file: " + input_file);
        }

        mup::ParserX p;
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

        std::string token;

        // Lambda to read next value/expression
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
                if constexpr (std::is_same_v<T, std::string>)
                {
                    var = token;
                    // Remove quotes if present
                    if (var.size() >= 2 && var.front() == '"' && var.back() == '"')
                    {
                        var = var.substr(1, var.size() - 2);
                    }
                }
                else
                {
                    try
                    {
                        p.SetExpr(token);
                        var = static_cast<T>(p.Eval().GetFloat());
                    }
                    catch (const mup::ParserError &e)
                    {
                        throw std::runtime_error("Parser Error parsing token '" + token + "': " + e.GetMsg());
                    }
                }
                return;
            }
            throw std::runtime_error("Error - Unexpected end of file.");
        };

        // 1. General Parameters
        next_value(num_runs);
        next_value(DimX);
        next_value(DimY);
        next_value(DimZ);
        next_value(Nx);
        next_value(Ny);
        next_value(Nz);
        next_value(dt);
        next_value(T);

        // ===========================================
        // 4. Calculate dx, dy, dz (Grid Spacing)
        // ===========================================
        if (Nx > 1)
        {
            dx = DimX / (Real)(Nx - 0.5);
        }
        else
        {
            dx = DimX;
        }
        if (Ny > 1)
        {
            dy = DimY / (Real)(Ny - 0.5);
        }
        else
        {
            dy = DimY;
        }
        if (Nz > 1)
        {
            dz = DimZ / (Real)(Nz - 0.5);
        }
        else
        {
            dz = DimZ;
        }

        // 2. Boundary Files
        next_value(u_boundary_file);
        next_value(p_boundary_file);

        // 3. Forcing Function Expressions
        next_value(fx_expression);
        next_value(fy_expression);
        next_value(fz_expression);

        // 4. Exact Solution Expressions
        next_value(u_exact_expression);
        next_value(v_exact_expression);
        next_value(w_exact_expression);
        next_value(p_exact_expression);

        // 5. Physics Parameters (Reading from your updated input file)
        next_value(k_expression); // Reads "1e10"
        next_value(nu);           // Reads 6.0

        // 6. Optional MPI Decomposition Parameters (default -1 for automatic)
        Px = -1;
        Py = -1;
        Pz = -1;
        try {
            next_value(Px);
            next_value(Py);
            next_value(Pz);
        } catch (...) {
            // Optional parameters not provided, use defaults
            Px = -1;
            Py = -1;
            Pz = -1;
        }

        std::cout << "\n===== Input Parameters Loaded =====\n";
        std::cout << "Forcing (Fx): " << fx_expression << "\n";
        std::cout << "Forcing (Fy): " << fy_expression << "\n";
        std::cout << "Forcing (Fz): " << fz_expression << "\n";
        std::cout << "Exact U:      " << u_exact_expression << "\n";
        std::cout << "Exact V:      " << v_exact_expression << "\n";
        std::cout << "Exact W:      " << w_exact_expression << "\n";
        std::cout << "Exact P:      " << p_exact_expression << "\n";
        std::cout << "K Function:   " << k_expression << "\n";
        std::cout << "Viscosity (nu): " << nu << "\n";
#ifdef USE_MPI
        std::cout << "MPI Decomposition: Px=" << Px << ", Py=" << Py << ", Pz=" << Pz << " (-1=auto)\n";
#endif
        std::cout << "===================================\n";
    }
};

// =========================================================================
// Helper Implementations
// =========================================================================

// 1. Forcing Function Helper
inline std::function<std::vector<Real>(Real, Real, Real, Real)> ParseInput::get_forcing_function() const
{
    std::string s_fx = fx_expression;
    std::string s_fy = fy_expression;
    std::string s_fz = fz_expression;

    return [s_fx, s_fy, s_fz](Real x, Real y, Real z, Real t) -> std::vector<Real>
    {
        mup::ParserX p;
        mup::Value vx((float)x), vy((float)y), vz((float)z), vt((float)t);
        p.DefineVar("x", mup::Variable(&vx));
        p.DefineVar("y", mup::Variable(&vy));
        p.DefineVar("z", mup::Variable(&vz));
        p.DefineVar("t", mup::Variable(&vt));
        try
        {
            p.DefineConst("pi", M_PI);
        }
        catch (...)
        {
        }

        std::vector<Real> res(3);
        try
        {
            p.SetExpr(s_fx);
            res[0] = (Real)p.Eval().GetFloat();
            p.SetExpr(s_fy);
            res[1] = (Real)p.Eval().GetFloat();
            p.SetExpr(s_fz);
            res[2] = (Real)p.Eval().GetFloat();
        }
        catch (...)
        {
            return {0, 0, 0};
        }
        return res;
    };
}

// 2. Exact Velocity Helper
inline std::function<std::vector<Real>(Real, Real, Real, Real)> ParseInput::get_exact_velocity_function() const
{
    std::string s_u = u_exact_expression;
    std::string s_v = v_exact_expression;
    std::string s_w = w_exact_expression;

    return [s_u, s_v, s_w](Real x, Real y, Real z, Real t) -> std::vector<Real>
    {
        mup::ParserX p;
        mup::Value vx((float)x), vy((float)y), vz((float)z), vt((float)t);
        p.DefineVar("x", mup::Variable(&vx));
        p.DefineVar("y", mup::Variable(&vy));
        p.DefineVar("z", mup::Variable(&vz));
        p.DefineVar("t", mup::Variable(&vt));
        try
        {
            p.DefineConst("pi", M_PI);
        }
        catch (...)
        {
        }

        std::vector<Real> res(3);
        try
        {
            p.SetExpr(s_u);
            res[0] = (Real)p.Eval().GetFloat();
            p.SetExpr(s_v);
            res[1] = (Real)p.Eval().GetFloat();
            p.SetExpr(s_w);
            res[2] = (Real)p.Eval().GetFloat();
        }
        catch (...)
        {
            return {0, 0, 0};
        }
        return res;
    };
}

// 3. Exact Pressure Helper
inline std::function<Real(Real, Real, Real, Real)> ParseInput::get_exact_pressure_function() const
{
    std::string s_p = p_exact_expression;

    return [s_p](Real x, Real y, Real z, Real t) -> Real
    {
        mup::ParserX p;
        mup::Value vx((float)x), vy((float)y), vz((float)z), vt((float)t);
        p.DefineVar("x", mup::Variable(&vx));
        p.DefineVar("y", mup::Variable(&vy));
        p.DefineVar("z", mup::Variable(&vz));
        p.DefineVar("t", mup::Variable(&vt));
        try
        {
            p.DefineConst("pi", M_PI);
        }
        catch (...)
        {
        }

        try
        {
            p.SetExpr(s_p);
            return (Real)p.Eval().GetFloat();
        }
        catch (...)
        {
            return 0.0;
        }
    };
}

// 4. Coefficient (K) Helper
inline std::function<Real(Real, Real, Real)> ParseInput::get_k_function() const
{
    std::string s_k = k_expression;

    return [s_k](Real x, Real y, Real z) -> Real
    {
        mup::ParserX p;
        mup::Value vx((float)x), vy((float)y), vz((float)z);
        p.DefineVar("x", mup::Variable(&vx));
        p.DefineVar("y", mup::Variable(&vy));
        p.DefineVar("z", mup::Variable(&vz));
        try
        {
            p.DefineConst("pi", M_PI);
        }
        catch (...)
        {
        }

        try
        {
            p.SetExpr(s_k);
            return (Real)p.Eval().GetFloat();
        }
        catch (...)
        {
            return 0.0;
        }
    };
}

#endif // PARSE_INPUT_HPP