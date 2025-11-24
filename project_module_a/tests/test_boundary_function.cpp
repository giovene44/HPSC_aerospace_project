#include "BoundaryFunctions.hpp"

// In your test file
template <Dim C>
Real evaluate_component(BoundaryFunctions &u, Real x, Real y, Real z, Real t)
{
    return u.value<C>(x, y, z, t);
}

Real get_value(BoundaryFunctions &u, Dim comp, Real x, Real y, Real z, Real t)
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

    return 0;
}