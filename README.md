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
