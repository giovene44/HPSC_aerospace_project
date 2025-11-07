#include "Variables.hpp"

template <typename StrideFunction>
struct DimensionsHandlerScalar
{
    Dim N1;
    Dim N2;
    Dim N3;
    Real dN1;
    StrideFunction stride;

    DimensionsHandlerScalar(Dim N1, Dim N2, Dim N3, Real dN1, StrideFunction stride)
        : N1(N1), N2(N2), N3(N3), dN1(dN1), stride(stride)
    {
    }
};

template <typename StrideFunction>
struct DimensionsHandlerVector
{
    Dim N1;
    Dim N2;
    Dim N3;
    Dim comp1;
    Dim comp2;
    Dim comp3;
    Real dN1;
    StrideFunction stride;

    DimensionsHandlerVector(Dim N1, Dim N2, Dim N3, Dim comp1, Dim comp2, Dim comp3, Real dN1, StrideFunction stride)
        : N1(N1), N2(N2), N3(N3), comp1(comp1), comp2(comp2), comp3(comp3), dN1(dN1), stride(stride)
    {
    }
};