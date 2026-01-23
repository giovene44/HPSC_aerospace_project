#!/usr/bin/env bash

make clean && 
make MPI=1 OMP=1 -j && 
OMP_NUM_THREADS=1  mpirun -n 1 ./bin/main_app &&
OMP_NUM_THREADS=2  mpirun -n 1 ./bin/main_app &&
OMP_NUM_THREADS=4  mpirun -n 1 ./bin/main_app &&
OMP_NUM_THREADS=8  mpirun -n 1 ./bin/main_app &&
OMP_NUM_THREADS=1  mpirun -n 2 ./bin/main_app &&
OMP_NUM_THREADS=2  mpirun -n 2 ./bin/main_app &&
OMP_NUM_THREADS=4  mpirun -n 2 ./bin/main_app &&
OMP_NUM_THREADS=8  mpirun -n 2 ./bin/main_app &&
OMP_NUM_THREADS=1  mpirun -n 4 ./bin/main_app &&
OMP_NUM_THREADS=2  mpirun -n 4 ./bin/main_app &&
OMP_NUM_THREADS=4  mpirun -n 4 ./bin/main_app && 
OMP_NUM_THREADS=1  mpirun -n 8 ./bin/main_app && 
OMP_NUM_THREADS=2  mpirun -n 8 ./bin/main_app &&
OMP_NUM_THREADS=1  mpirun -n 12 ./bin/main_app