#ifndef DIMENSIONHANDLER_HPP
#define DIMENSIONHANDLER_HPP

#include "Variables.hpp"

template <typename StrideFunction>
struct DimensionsHandlerScalar
{
    Dim N1;
    Dim N2;
    Dim N3;
    Real dN1;
    StrideFunction stride;

    constexpr DimensionsHandlerScalar(Dim N1, Dim N2, Dim N3, Real dN1, StrideFunction stride) noexcept
        : N1(N1), N2(N2), N3(N3), dN1(dN1), stride(stride)
    {
    }
};

template <typename StrideFunction>
struct DimensionsHandlerVector : public DimensionsHandlerScalar<StrideFunction>
{
    Dim Comp1;
    Dim Comp2;
    Dim Comp3;

    constexpr DimensionsHandlerVector(Dim N1, Dim N2, Dim N3, Dim Comp1, Dim Comp2, Dim Comp3, Real dN1, StrideFunction stride) noexcept
        : DimensionsHandlerScalar<StrideFunction>(N1, N2, N3, dN1, stride), Comp1(Comp1), Comp2(Comp2), Comp3(Comp3)
    {
    }
};

#endif // DIMENSIONHANDLER_HPP