#include "variables.hpp"

template <typename StrideFunction>
struct DimensionsHandler
{
    Dim N1;
    Dim N2;
    Dim N3;
    Real dN1;
    StrideFunction stride;

    DimensionsHandler(Dim N1, Dim N2, Dim N3, Real dN1, StrideFunction stride)
        : N1(N1), N2(N2), N3(N3), dN1(dN1), stride(stride)
    {
    }
};