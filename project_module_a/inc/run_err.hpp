#ifndef MMS_TEST_RUNNER_HPP
#define MMS_TEST_RUNNER_HPP

/**
 * @brief Convergence study with the method of manufactured solutions.
 *
 * Reads the input file (default ./Input/Input.in), runs num_runs simulations doubling the
 * number of grid points every time (and dividing dt by dt_refinement), and writes the
 * errors and the timings to OUTPUT/Convergence_Analysis_*.dat.
 *
 * Command-line options:
 *   -i <file>          input file
 *   --procs Px Py Pz   MPI process grid (default: chosen by MPI_Dims_create)
 *   --no-omp-ref       skip the single-thread reference run used for the OpenMP speed-up
 *   --vtk              write the final solution of every run (OUTPUT/solution_N*_rank*.vtk)
 */
int run_convergence(int argc, char **argv);

#ifdef USE_MPI
int run_multiple_mpi(int argc, char **argv);
#else
int run_multiple(int argc = 0, char **argv = nullptr);
#endif

#endif // MMS_TEST_RUNNER_HPP
