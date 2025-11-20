#include "BoundaryFunctions.hpp"

// In your test file
template<Dim C>
Real evaluate_component(BoundaryFunctions& u, Real x, Real y, Real z, Real t)
{
    return u.value<C>(x, y, z, t);
}

Real get_value(BoundaryFunctions& u, Dim comp, Real x, Real y, Real z, Real t)
{
    if (comp == 0)
        return evaluate_component<0>(u, x, y, z, t);
    else if (comp == 1)
        return evaluate_component<1>(u, x, y, z, t);
    else if (comp == 2)
        return evaluate_component<2>(u, x, y, z, t);
    else
        throw std::runtime_error("Invalid component");
}

int main()
{
    BoundaryFunctions p;
    p.setParsing("./Input/boundary_p0.dat");

    auto p_ex = [](Real x, Real y, Real z, Real t)
    {
        return (-3.0f / 100.0) * std::cos(x) * std::sin(y) * (std::sin(z) - std::cos(z));
    };

    bool error_found = false;
    for (Real x = 0.0; x <= 1.0; x += 0.05)
    {
        for (Real y = 0.0; y <= 1.0; y += 0.05)
        {
            for (Real z = 0.0; z <= 1.0; z += 0.05)
            {
                Real t = 0.0;
                Real p_val = p.value<0>(x, y, z, t);
                Real p_exact = p_ex(x, y, z, t);
                if (std::abs(p_val - p_exact) > 0)
                {
                    error_found = true;
                    std::cout << "x: " << x << " y: " << y << " z: " << z
                              << "\t|p_val: " << p_val << "\t|p_exact: " << p_exact
                              << " \t|error: " << std::abs(p_val - p_exact) << std::endl;
                }
            }
        }
    }
    if (!error_found)
    {
        std::cout << "All boundary function values for pressure are within the acceptable error margin." << std::endl;
    }

    BoundaryFunctions u;
    u.setParsing("./Input/boundary_u0.dat");
    auto u_ex = [](Real x, Real y, Real z, Real t, Dim component) -> Real
    {
        if (component == 0)
            return std::sin(t) * std::sin(x) * std::sin(y) * std::sin(z);
        else if (component == 1)
            return std::sin(t) * std::cos(x) * std::cos(y) * std::cos(z);
        else if (component == 2)
            return std::sin(t) * std::cos(x) * std::sin(y) * (std::sin(z) + std::cos(z));
        else
            throw std::runtime_error("u_ex: invalid component index");
    };

    error_found = false;
    for (Real x = 0.0; x <= 1.0; x += 0.05)
    {
        for (Real y = 0.0; y <= 1.0; y += 0.05)
        {
            for (Real z = 0.0; z <= 1.0; z += 0.05)
            {
                for (Real t = 0.0; t <= 1.0; t += 0.05)
                {
                    for (Dim comp = 0; comp < 3; comp++)
                    {
                        Real u_val = get_value(u, comp, x, y, z, t);
                        Real u_exact = u_ex(x, y, z, t, comp);
                        if (std::abs(u_val - u_exact) > 0)
                        {
                            error_found = true;
                            std::cout << "Component: " << comp
                                      << " x: " << x << " y: " << y << " z: " << z
                                      << "\t|u_val: " << u_val << "\t|u_exact: " << u_exact
                                      << " \t|error: " << std::abs(u_val - u_exact) << std::endl;
                        }
                    }
                }
            }
        }
    }
    if (!error_found)
    {
        std::cout << "All boundary function values for velocity are within the acceptable error margin." << std::endl;
    }
    return 0;
}