#ifndef BOUNDARY_FUNCTIONS_HPP
#define BOUNDARY_FUNCTIONS_HPP
#include "Variables.hpp"
#include "mpParser.h"
#include <string>
#include <fstream>

using namespace mup;

class BoundaryFunctions
{
public:
    void setParsing(std::string input_file)
    {
        std::ifstream file(input_file);
        if (!file.is_open())
        {
            throw std::runtime_error("Unable to open file: " + input_file);
        }
        std::string line;
        // std::cout<<"File is open: " << input_file << std::endl;
        while (std::getline(file, line))
        {
            if (line[0] == '#') // Skip comment lines
                continue;
            string_expression.emplace_back(line);
            // std::cout<<"Read line: " << line << std::endl;
        }
        // std::cout<<"Get all line " << std::endl;
    }

    template <Dim component = 0>
    Real value(Real x_, Real y_, Real z_, Real t_) const
    {
        // Thread-local parser and values: each thread gets its own instance
        thread_local ParserX p_local;
        thread_local Value xval_local;
        thread_local Value yval_local;
        thread_local Value zval_local;
        thread_local Value tval_local;
        thread_local bool initialized = false;

        if (!initialized)
        {
            xval_local = Value(Real(0.0));
            yval_local = Value(Real(0.0));
            zval_local = Value(Real(0.0));
            tval_local = Value(Real(0.0));
            p_local.DefineVar("x", Variable(&xval_local));
            p_local.DefineVar("y", Variable(&yval_local));
            p_local.DefineVar("z", Variable(&zval_local));
            p_local.DefineVar("t", Variable(&tval_local));
            initialized = true;
        }

        try
        {
            p_local.SetExpr(string_expression[component]);
        }
        catch (ParserError &e)
        {
            std::cerr << "Parser error in BoundaryFunctions::value(): " << e.GetMsg() << std::endl;
            throw;
        }
        xval_local = Value(x_);
        yval_local = Value(y_);
        zval_local = Value(z_);
        tval_local = Value(t_);

        return p_local.Eval().GetFloat();
    }

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

        return (value<component>(xp, yp, zp, t_) -
                value<component>(xm, ym, zm, t_)) /
               (d);
    }

    void set_string_expression(const std::vector<std::string> &exprs)
    {
        string_expression = exprs;
    }

private:
    std::vector<std::string> string_expression;
};

#endif // BOUNDARY_FUNCTIONS_HPP