# HPSC_aerospace_project

## Project Overview

This project implements and benchmarks parallel and serial solvers for tridiagonal systems, focusing on the Schur complement method (MPI-based) and the Thomas algorithm (serial). The main application is solving 1D/3D PDEs relevant to aerospace simulations, with strong and weak scalability analysis. The codebase is designed for the High Performance Scientific Computing for Aerospace course at Politecnico di Milano.

### Key Features

- Parallel solver using MPI and OpenMP (Schur complement method)
- Serial solver using the Thomas algorithm
- Automated benchmarking and speedup analysis
- Modular structure for easy extension to more complex PDEs

## How to Run

To build and run the project with various MPI process and OpenMP thread configurations, use the provided `run.sh` script:

```bash
cd project_module_a
./run.sh
```

This script will:

- Clean and rebuild the project with MPI and OpenMP enabled
- Execute the main application (`main_app`) with different combinations of MPI processes (`-n`) and OpenMP threads (`OMP_NUM_THREADS`)

Results and outputs will be generated in the appropriate output folders for further analysis and plotting.

### Building by hand

```bash
cd project_module_a
make                  # serial
make OMP=1            # OpenMP (on macOS it uses Homebrew's libomp: brew install libomp)
make MPI=1 OMP=1      # MPI + OpenMP
make MPI=1 OMP=1 check CHECK_NP=8   # automatic checks (tridiagonal solver, MMS convergence)
OMP_NUM_THREADS=4 mpirun --bind-to none -n 2 ./bin/main_app [-i Input/Input.in] [--procs Px Py Pz] [--no-omp-ref] [--vtk]
```

The analytic expressions of the input file (forcing, exact solution, permeability) are compiled
by `inc/Expression.hpp`, so muparserx is no longer needed. `Input/Input.in` also accepts four
optional trailing entries: the initial time `t0`, the factor dividing `dt` at every run
(`2` = `dt` proportional to `h`), the factor multiplying `N` at every run (`1` = temporal
study on a fixed grid) and `chi`, the rotational pressure correction (`0` = standard
incremental scheme, `1` = rotational form, second-order pressure).
The pressure error is computed up to an additive constant (Dirichlet velocity on the whole
boundary) and against `p(t - dt/2)`, since the pressure is staggered in time.
