#include "Variables.hpp"

struct DimensionsHandlerScalar
{
    Dim N1;
    Dim N2;
    Dim N3;
    Real dN1;

    constexpr DimensionsHandlerScalar(Dim N1, Dim N2, Dim N3, Real dN1) noexcept
        : N1(N1), N2(N2), N3(N3), dN1(dN1)
    {
    }
};

struct DimensionsHandlerVector : public DimensionsHandlerScalar
{
    Dim Comp1;
    Dim Comp2;
    Dim Comp3;

    constexpr DimensionsHandlerVector(Dim N1, Dim N2, Dim N3, Dim Comp1, Dim Comp2, Dim Comp3, Real dN1) noexcept
        : DimensionsHandlerScalar(N1, N2, N3, dN1), Comp1(Comp1), Comp2(Comp2), Comp3(Comp3)
    {
    }
};