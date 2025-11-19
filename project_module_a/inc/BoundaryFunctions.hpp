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
        xval = Value(Real(0.0));
        yval = Value(Real(0.0));
        zval = Value(Real(0.0));
        tval = Value(Real(0.0));

        p.DefineVar("x", Variable(&xval));
        p.DefineVar("y", Variable(&yval));
        p.DefineVar("z", Variable(&zval));
        p.DefineVar("t", Variable(&tval));
    }

    template <Dim component = 0>
    Real value(Real x_, Real y_, Real z_, Real t_)
    {
        if constexpr (component == 0)
            p.SetExpr(string_expression[0]);
        else if constexpr (component == 1)
            p.SetExpr(string_expression[1]);
        else if constexpr (component == 2)
            p.SetExpr(string_expression[2]);
        else
            throw std::runtime_error("BoundaryFunctions::value: invalid component index");
        xval = Value(x_);
        yval = Value(y_);
        zval = Value(z_);
        tval = Value(t_);

        return p.Eval().GetFloat();
    }

    template <Dim component = 0>
    Real first_derivative(Real x_, Real y_, Real z_, Real t_, Real d)
    {
        return (value<component>(x_ + d, y_, z_, t_) - value<component>(x_ - d, y_, z_, t_)) / (2.0 * d);
    }

private:
    ParserX p;
    std::vector<std::string> string_expression;
    Value xval;
    Value yval;
    Value zval;
    Value tval;
};

#endif // BOUNDARY_FUNCTIONS_HPP