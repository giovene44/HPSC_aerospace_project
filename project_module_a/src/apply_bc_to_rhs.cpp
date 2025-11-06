#include "ScalarVariable.hpp"
#include "NavierStokesBrinkman.hpp"

// CODE TO BE LATER INSERTED INTO THE MAIN CLASS NavierStokesBrinkman

// Assuming global variables for grid dimensions and spacing
extern Dim Nx;
extern Dim Ny;
extern Dim Nz;
extern Real dx;
extern Real dy;
extern Real dz;


std::function<float(float, float, float)> BC_function_u = [](float x, float y, float z) {
    // Example boundary condition function: Dirichlet BC with value 1.0
    return 1.0f;
};

std::function<float(float, float, float)> BC_function_v = [](float x, float y, float z) {
    // Example boundary condition function: Dirichlet BC with value 1.0
    return 1.0f;
};

std::function<float(float, float, float)> BC_function_w = [](float x, float y, float z) {
    // Example boundary condition function: Dirichlet BC with value 1.0
    return 1.0f;
};

void apply_bc_to_rhs1(ScalarVariable &rhs)
{
    //applying boundary conditions to rhs_1
    
    // for points corresponding to first row of the block (i=0) i use divergence technique:

    for (Dim j=0; j<Ny; j++){
        for(Dim k=0; k<Nz; k++){

            float dv_dy = BC_function_v(0.0, (j+1/2)*dy, k*dz)-BC_function_v(0.0, (j-1/2)*dy, k*dz);
            float du_dx = BC_function_w(0.0, j*dy, (k+1/2)*dz)-BC_function_w(0.0, j*dy, (k-1/2)*dz);
            rhs.set(0,j,k) =BC_function_u(0.0, j*dy, k*dz) + dx/2 * (-dv_dy - du_dx);


            rhs.set(Nx-1,j,k)= BC_function_u((Nx-1)*dx, j*dy, k*dz);
        }
    }
}

void apply_bc_to_rhs2(ScalarVariable &rhs)
{
    //applying boundary conditions to rhs_2
    
    // for points corresponding to last row of the block (i=Nx-1) i use ghost node technique:

    for (Dim j=0; j<Ny; j++){
        for(Dim k=0; k<Nz; k++){

            rhs.set(0,j,k) = BC_function_v(0.0, j*dy, k*dz);


            float c = -gamma.get(Nx-1,j,k);  // I noticed that gamma is actually defined on the pressure nodes, not on velocity nodes! this is not exactly right but maybe acceptable for now
            rhs.set(Nx-1,j,k)= rhs.get(Nx-1,j,k) - 2.0*c* BC_function_v((Nx-1)*dx, j*dy, k*dz);
        }
    }
}

void apply_bc_to_rhs3(ScalarVariable &rhs)
{
    //applying boundary conditions to rhs_3

    //Again, for points corresponding to last row of the block (i=Nx-1) i use ghost node technique:

    for (Dim j=0; j<Ny; j++){
        for(Dim k=0; k<Nz; k++){

            rhs.set(0,j,k) = BC_function_w(0.0, j*dy, k*dz);


            float c = -gamma.get(Nx-1,j,k);
            rhs.set(Nx-1,j,k)= rhs.get(Nx-1,j,k) - 2.0*c* BC_function_w((Nx-1)*dx, j*dy, k*dz);
        }
    }
}

void apply_bc_to_rhs4(ScalarVariable &rhs)
{
    //applying boundary conditions to rhs_1 for the rhs used when solving along y!

    // for points corresponding to last row of the block (j=Ny-1) i use ghost node technique:

    for (Dim i=0; i<Nx; j++){
        for(Dim k=0; k<Nz; k++){
            
            rhs.set(i,0,k) = BC_function_u(i*dx, 0.0, k*dz);
            float c = -gamma.get(i,Ny-1,k);
            rhs.set(i,Ny-1,k)= rhs.get(i,Ny-1,k) - 2.0*c* BC_function_u(i*dx, (Ny-1)*dy, k*dz);

        }
    }
}

void apply_bc_to_rhs5(ScalarVariable &rhs)
{
    //applying boundary conditions to rhs_2 for the rhs used when solving along y!

    // for points corresponding to first row of the block (j=0) i use divergence technique:

    for (Dim i=0; i<Nx; j++){
        for(Dim k=0; k<Nz; k++){

            float du_dx = BC_function_u((i+1/2)*dx, 0.0, k*dz)-BC_function_u((i-1/2)*dx, 0.0, k*dz);
            float dw_dz = BC_function_w(i*dx, 0.0, (k+1/2)*dz)-BC_function_w(i*dx, 0.0, (k-1/2)*dz);
            rhs.set(i,0,k) =BC_function_v(i*dx, 0.0, k*dz) + dy/2 * (-du_dx - dw_dz);

            rhs.set(i,Ny-1,k)= BC_function_v(i*dx, (Ny-1)*dy, k*dz);
        }
    }
}

void apply_bc_to_rhs6(ScalarVariable &rhs)
{
    //applying boundary conditions to rhs_3 for the rhs used when solving along y!

    //Again, for points corresponding to last row of the block (j=Ny-1) i use ghost node technique:

    for (Dim i=0; i<Nx; j++){
        for(Dim k=0; k<Nz; k++){

            rhs.set(i,0,k) = BC_function_w(i*dx, 0.0, k*dz);
            float c = -gamma.get(i,Ny-1,k);
            rhs.set(i,Ny-1,k)= rhs.get(i,Ny-1,k) - 2.0*c* BC_function_w(i*dx, (Ny-1)*dy, k*dz);
        }
    }
}


void apply_bc_to_rhs7(ScalarVariable &rhs)
{
    //applying boundary conditions to rhs_1 for the rhs used when solving along z!

    // for points corresponding to last row of the block (k=Nz-1) i use ghost node technique:

    for (Dim i=0; i<Nx; j++){
        for(Dim j=0; j<Ny; j++){

            rhs.set(i,j,0) = BC_function_u(i*dx, j*dy, 0.0);
            float c = -gamma.get(i,j,Nz-1);
            rhs.set(i,j,Nz-1)= rhs.get(i,j,Nz-1) - 2.0*c* BC_function_u(i*dx, j*dy, (Nz-1)*dz);

        }
    }
}

void apply_bc_to_rhs8(ScalarVariable &rhs)
{
    //applying boundary conditions to rhs_2 for the rhs used when solving along z!

    // for points corresponding to last row of the block (k=Nz-1) i use ghost node technique:

    for (Dim i=0; i<Nx; j++){
        for(Dim j=0; j<Ny; j++){

            rhs.set(i,j,0) = BC_function_v(i*dx, j*dy, 0.0);
            float c = -gamma.get(i,j,Nz-1);
            rhs.set(i,j,Nz-1)= rhs.get(i,j,Nz-1) - 2.0*c* BC_function_v(i*dx, j*dy, (Nz-1)*dz);

        }
    }
}

void apply_bc_to_rhs9(ScalarVariable &rhs)
{
    //applying boundary conditions to rhs_3 for the rhs used when solving along z!

    // for points corresponding to first row of the block (k=0) i use divergence technique:

    for (Dim i=0; i<Nx; j++){
        for(Dim j=0; j<Ny; j++){

            float du_dx = BC_function_u((i+1/2)*dx, j*dy, 0.0)-BC_function_u((i-1/2)*dx, j*dy, 0.0);
            float dv_dy = BC_function_v(i*dx, (j+1/2)*dy, 0.0)-BC_function_v(i*dx, (j-1/2)*dy, 0.0);
            rhs.set(i,j,0) =BC_function_w(i*dx, j*dy, 0.0) + dz/2 * (-du_dx - dv_dy);

            rhs.set(i,j,Nz-1)= BC_function_w(i*dx, j*dy, (Nz-1)*dz);
        }
    }
}