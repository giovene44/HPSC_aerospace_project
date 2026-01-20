#ifndef MMS_TEST_RUNNER_HPP
#define MMS_TEST_RUNNER_HPP
#include <iostream>
#include <cmath>
#include <cstdlib> // for system()
#include <utility> // for std::pair
#include <fstream>

#include "manufactured_solution_technique.hpp"
#include "navier_stokes_brinkman.hpp"
#include "ScalarVariable.hpp"
#include "VectorVariable.hpp"

#ifdef USE_MPI
#include "MPICommunicator.hpp"
#include "MPITopology3D.hpp"
int run_multiple_mpi(int argc, char **argv);
#else
int run_multiple(bool use_openMP = false);
#endif

#endif // MMS_TEST_RUNNER_HPP