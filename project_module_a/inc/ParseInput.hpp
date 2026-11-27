#ifndef PARSE_INPUT_HPP
#define PARSE_INPUT_HPP

#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Variables.hpp"
#include "Expression.hpp"
#include "BoundaryFunctions.hpp"

/**
 * @brief Reader of the input file (see Input/Input.in for the format).
 *
 * Mandatory entries, in order (lines starting with '#' are comments):
 *   number of runs, Lx, Ly, Lz, Nx Ny Nz, dt, T, velocity boundary file, pressure boundary
 *   file, fx, fy, fz, exact u, v, w, exact p, permeability k(x,y,z), viscosity nu.
 * Optional trailing entries (backward compatible):
 *   t0                 initial time (default 1.5, the value used so far),
 *   dt_refinement      factor dividing dt at every refinement level (default 1: fixed dt;
 *                      use 2 to refine dt together with the grid spacing),
 *   grid_refinement    factor multiplying N at every refinement level (default 2; use 1 for
 *                      a temporal convergence study on a fixed grid),
 *   chi                rotational pressure correction, 0 <= chi <= 1 (default 0: standard
 *                      incremental scheme of the course; 1 reduces the pressure error).
 * Numeric entries may be expressions (e.g. 2.0*pi).
 */
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
    int num_runs = 1;
    Real DimX = 1, DimY = 1, DimZ = 1;
    Dim Nx = 8, Ny = 8, Nz = 8;
    Real dt = 0.1, T = 1.0;
    Real dx = 1, dy = 1, dz = 1;
    Real t0 = 1.5;
    Real dt_refinement = 1.0;
    int grid_refinement = 2;
    Real chi = 0.0;

    std::string u_boundary_file;
    std::string p_boundary_file;

    std::string fx_expression, fy_expression, fz_expression;
    std::string u_exact_expression, v_exact_expression, w_exact_expression, p_exact_expression;

    std::string k_expression;
    Real nu = 1.0;

    // --- Helpers to generate std::function objects / BoundaryFunctions ---
    std::function<std::vector<Real>(Real, Real, Real, Real)> get_forcing_function() const
    {
        return vector_function(fx_expression, fy_expression, fz_expression);
    }
    BoundaryFunctions get_forcing_function_bf() const
    {
        BoundaryFunctions f;
        f.set_string_expression({fx_expression, fy_expression, fz_expression});
        return f;
    }
    std::function<std::vector<Real>(Real, Real, Real, Real)> get_exact_velocity_function() const
    {
        return vector_function(u_exact_expression, v_exact_expression, w_exact_expression);
    }
    BoundaryFunctions get_exact_velocity_function_bf() const
    {
        BoundaryFunctions f;
        f.set_string_expression({u_exact_expression, v_exact_expression, w_exact_expression});
        return f;
    }
    std::function<Real(Real, Real, Real, Real)> get_exact_pressure_function() const
    {
        fastexpr::Expression e(p_exact_expression);
        return [e](Real x, Real y, Real z, Real t) -> Real
        { return e(x, y, z, t); };
    }
    BoundaryFunctions get_exact_pressure_function_bf() const
    {
        BoundaryFunctions f;
        f.set_string_expression({p_exact_expression});
        return f;
    }
    std::function<Real(Real, Real, Real)> get_k_function() const
    {
        fastexpr::Expression e(k_expression);
        return [e](Real x, Real y, Real z) -> Real
        { return e(x, y, z, 0.0); };
    }
    BoundaryFunctions get_k_function_bf() const
    {
        BoundaryFunctions f;
        f.set_string_expression({k_expression});
        return f;
    }

    void parse_input(const std::string &input_file)
    {
        std::ifstream file(input_file);
        if (!file.is_open())
            throw std::runtime_error("Error - Cannot open input file: " + input_file);

        // One entry per non-empty line; '#' starts a comment that lasts until end of line.
        // (Expressions may contain spaces; only the "Nx Ny Nz" line is split.)
        std::vector<std::string> lines;
        {
            std::string line;
            while (std::getline(file, line))
            {
                const auto hash = line.find('#');
                if (hash != std::string::npos)
                    line = line.substr(0, hash);
                const auto first = line.find_first_not_of(" \t\r");
                if (first == std::string::npos)
                    continue;
                const auto last = line.find_last_not_of(" \t\r");
                lines.push_back(line.substr(first, last - first + 1));
            }
        }
        std::vector<std::string> tokens;
        for (std::size_t q = 0; q < lines.size(); ++q)
        {
            if (q == 4) // Nx Ny Nz
            {
                std::istringstream ss(lines[q]);
                std::vector<std::string> n3;
                std::string tok;
                while (ss >> tok)
                    n3.push_back(tok);
                if (n3.size() == 1)
                    n3 = {n3[0], n3[0], n3[0]};
                if (n3.size() != 3)
                    throw std::runtime_error("Error - expected 'Nx Ny Nz' in " + input_file + ", got: " + lines[q]);
                tokens.insert(tokens.end(), n3.begin(), n3.end());
            }
            else
                tokens.push_back(lines[q]);
        }
        std::size_t pos = 0;
        auto next_token = [&](const char *what) -> std::string
        {
            if (pos >= tokens.size())
                throw std::runtime_error(std::string("Error - Unexpected end of input file while reading ") + what);
            std::string t = tokens[pos++];
            if (t.size() >= 2 && t.front() == '"' && t.back() == '"')
                t = t.substr(1, t.size() - 2);
            return t;
        };
        auto next_number = [&](const char *what) -> double
        {
            const std::string t = next_token(what);
            fastexpr::Expression e(t);
            if (!e.is_constant())
                throw std::runtime_error(std::string("Error - '") + t + "' is not a constant (" + what + ")");
            return e(0, 0, 0, 0);
        };

        num_runs = static_cast<int>(next_number("number of runs"));
        DimX = next_number("Lx");
        DimY = next_number("Ly");
        DimZ = next_number("Lz");
        Nx = static_cast<Dim>(next_number("Nx"));
        Ny = static_cast<Dim>(next_number("Ny"));
        Nz = static_cast<Dim>(next_number("Nz"));
        dt = next_number("dt");
        T = next_number("T");

        // Staggered grid of the course: pressure nodes at i*h, i = 0..N-1, and the domain is
        // [0, (N - 1/2) h], so that the last normal-velocity node lies on the boundary.
        dx = DimX / (Real(Nx) - Real(0.5));
        dy = DimY / (Real(Ny) - Real(0.5));
        dz = DimZ / (Real(Nz) - Real(0.5));

        u_boundary_file = next_token("velocity boundary file");
        p_boundary_file = next_token("pressure boundary file");

        fx_expression = next_token("fx");
        fy_expression = next_token("fy");
        fz_expression = next_token("fz");

        u_exact_expression = next_token("exact u");
        v_exact_expression = next_token("exact v");
        w_exact_expression = next_token("exact w");
        p_exact_expression = next_token("exact p");

        k_expression = next_token("permeability k");
        nu = next_number("nu");

        if (pos < tokens.size())
            t0 = next_number("t0");
        if (pos < tokens.size())
            dt_refinement = next_number("dt refinement");
        if (pos < tokens.size())
            grid_refinement = static_cast<int>(next_number("grid refinement"));
        if (pos < tokens.size())
            chi = next_number("chi (rotational correction)");

        // validate the expressions immediately (clear error messages)
        for (const std::string *s : {&fx_expression, &fy_expression, &fz_expression, &u_exact_expression,
                                     &v_exact_expression, &w_exact_expression, &p_exact_expression, &k_expression})
            fastexpr::Expression check(*s);
        if (num_runs < 1 || Nx < 3 || Ny < 3 || Nz < 3 || dt <= 0 || T < 0 || nu <= 0 || dt_refinement <= 0 ||
            grid_refinement < 1 || chi < 0 || chi > 1)
            throw std::runtime_error("Error - invalid parameters in " + input_file);
    }

private:
    static std::function<std::vector<Real>(Real, Real, Real, Real)>
    vector_function(const std::string &a, const std::string &b, const std::string &c)
    {
        fastexpr::Expression ea(a), eb(b), ec(c);
        return [ea, eb, ec](Real x, Real y, Real z, Real t) -> std::vector<Real>
        { return {ea(x, y, z, t), eb(x, y, z, t), ec(x, y, z, t)}; };
    }
};

#endif // PARSE_INPUT_HPP
