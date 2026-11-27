#!/usr/bin/env bash
# Build the MPI + OpenMP solver and run the convergence study (Input/Input.in) for several
# combinations of MPI processes and OpenMP threads. Results: OUTPUT/Convergence_Analysis_MPI_<P>_OpenMP_<T>.dat
#
# MPIRUN_FLAGS: extra options for mpirun. With Open MPI, "--bind-to none" lets the OpenMP
# threads of a rank use more than one core (the default binding pins each rank to one core).
set -e
MPIRUN_FLAGS=${MPIRUN_FLAGS:-"--bind-to none --oversubscribe"}

make clean
make MPI=1 OMP=1 -j

run() { # processes threads
    echo "=== mpirun -n $1, OMP_NUM_THREADS=$2"
    OMP_NUM_THREADS=$2 mpirun $MPIRUN_FLAGS -n "$1" ./bin/main_app
}

run 1 1
run 1 2
run 1 4
run 1 8
run 2 1
run 2 2
run 2 4
run 2 8
run 4 1
run 4 2
run 4 4
run 8 1
run 8 2
run 8 4
run 12 1
