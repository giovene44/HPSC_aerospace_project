#ifndef BOUNDARY_FUNCTIONS_HPP
#define BOUNDARY_FUNCTIONS_HPP

#include "Variables.hpp"
#include "Expression.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @brief A set of analytic functions f_c(x, y, z, t) (one per component) read from a file
 *        or given as strings. Expressions are compiled once (see Expression.hpp), the
 *        evaluation is thread safe and does not allocate.
 */
class BoundaryFunctions
{
public:
    /// Read one expression per line; empty lines and lines starting with '#' are skipped.
    void setParsing(const std::string &input_file)
    {
        std::ifstream file(input_file);
        if (!file.is_open())
            throw std::runtime_error("Unable to open file: " + input_file);
        std::vector<std::string> exprs;
        std::string line;
        while (std::getline(file, line))
        {
            const auto first = line.find_first_not_of(" \t\r");
            if (first == std::string::npos || line[first] == '#')
                continue;
            const auto last = line.find_last_not_of(" \t\r");
            exprs.emplace_back(line.substr(first, last - first + 1));
        }
        set_string_expression(exprs);
    }

    void set_string_expression(const std::vector<std::string> &exprs)
    {
        string_expression = exprs;
        expr_.clear();
        expr_.reserve(exprs.size());
        for (const auto &s : exprs)
            expr_.emplace_back(s);
    }

    std::size_t size() const { return expr_.size(); }
    const std::string &string_of(int component) const { return string_expression.at(component); }
    const fastexpr::Expression &expression(int component) const { return expr_.at(component); }

    template <Dim component = 0>
    Real value(Real x_, Real y_, Real z_, Real t_) const
    {
        return expr_[component](x_, y_, z_, t_);
    }

    Real value(int component, Real x_, Real y_, Real z_, Real t_) const
    {
        return expr_[component](x_, y_, z_, t_);
    }

    /// Centred finite difference (step d) of component `component` along its own direction.
    template <Dim component = 0>
    Real first_derivative(Real x_, Real y_, Real z_, Real t_, Real d) const
    {
        Real xp = x_, yp = y_, zp = z_;
        Real xm = x_, ym = y_, zm = z_;
        if constexpr (component == 0)
        {
            xp += d * Real(0.5);
            xm -= d * Real(0.5);
        }
        else if constexpr (component == 1)
        {
            yp += d * Real(0.5);
            ym -= d * Real(0.5);
        }
        else if constexpr (component == 2)
        {
            zp += d * Real(0.5);
            zm -= d * Real(0.5);
        }
        return (value<component>(xp, yp, zp, t_) - value<component>(xm, ym, zm, t_)) / d;
    }

private:
    std::vector<std::string> string_expression;
    std::vector<fastexpr::Expression> expr_;
};

#endif // BOUNDARY_FUNCTIONS_HPP
