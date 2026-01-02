#!/bin/bash

# Stop execution immediately if any command fails
set -e

echo "--- Cleaning Build ---"
make clean

echo "--- Compiling Tests ---"
make test -j OMP=1

echo
echo "--- Running Direction-Splitting Validation Tests ---"

echo
echo "X-direction split test:"
echo "  Manufactured solution u(x,y,z,t)"
echo "  PDE: u_t - nu * u_xx = f"
echo "  Time step: (I - gamma * D_xx) u^{n+1} = u^n + dt*f + gamma*D_xx u^n"
echo "  Boundary conditions: lecture staggered BCs + known faces"
OMP_NUM_THREADS=8 OMP_PROC_BIND=close ./bin/test_navier_stokes_x_direction

echo
echo "Y-direction split test:"
echo "  Manufactured solution u(x,y,z,t)"
echo "  PDE: u_t - nu * u_yy = f"
echo "  Time step: (I - gamma * D_yy) u^{n+1} = u^n + dt*f + gamma*D_yy u^n"
echo "  Boundary conditions: lecture staggered BCs + known faces"
OMP_NUM_THREADS=8 OMP_PROC_BIND=close ./bin/test_navier_stokes_y_direction

echo
echo "Z-direction split test:"
echo "  Manufactured solution u(x,y,z,t)"
echo "  PDE: u_t - nu * u_zz = f"
echo "  Time step: (I - gamma * D_zz) u^{n+1} = u^n + dt*f + gamma*D_zz u^n"
echo "  Boundary conditions: lecture staggered BCs + known faces"
OMP_NUM_THREADS=8 OMP_PROC_BIND=close ./bin/test_navier_stokes_z_direction
