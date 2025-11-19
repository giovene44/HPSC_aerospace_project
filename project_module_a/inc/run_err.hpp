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

std::pair<Real, Real> single_run(Real N, Real dt);

int run_multiple();

#endif // MMS_TEST_RUNNER_HPP