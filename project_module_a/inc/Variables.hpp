#ifndef DEF_VARIABLE_HPP
#define DEF_VARIABLE_HPP

// Double precision is required for meaningful convergence studies: with single precision
// the velocity increment over one time step (O(dt)) keeps only a few significant digits
// and the pressure increment (divergence / dt) is dominated by round-off.
using Real = double;

using Dim = int;

#ifdef USE_MPI
#include <mpi.h>
#include <type_traits>
/// MPI datatype matching Real.
inline MPI_Datatype mpi_real_type()
{
    return std::is_same<Real, double>::value ? MPI_DOUBLE : MPI_FLOAT;
}
#endif

#endif // DEF_VARIABLE_HPP
