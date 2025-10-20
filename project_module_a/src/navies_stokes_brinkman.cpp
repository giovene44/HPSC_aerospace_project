#include "navier_stokes_brinkmann.hpp"
#include "VectorVariable.hpp"

void NavierStokesBrinkmann::assemble()


void NavierStokesBrinkmann::assemble_g_rhs()
{

    //this method assembles the g vector for the current time step.
    //(g vector is the rhs of the momentum equation!)

    double t = 0.0; // Placeholder for current time, (TODO) should be set appropriately

    for (int i=0; i<g_rhs.size(); i++){ //i indicates the component among the 3 components of the vector
        for (int j=0; j<g_rhs[0].size(); j++) //j indicates the index of the element in the vector
        {

            float eta_local_second_derivative = eta.second_derivative(i, 0, j); // x direction
            float zeta_local_second_derivative = zeta.second_derivative(i, 1, j); // y direction
            float u_local_second_derivative = u.second_derivative(i, 2, j); // z direction

            g_rhs[i][j] = forcing_term.value(i,j,t) - gradient_pressure[i][j] 
            - eta*velocity_predictor[i][j]/(2.0* /*K[i][j] or K.get(i,j)*/)+ eta/2.0 
            * (eta_local_second_derivative + zeta_local_second_derivative + u_local_second_derivative);

        }
    }
}

void NavierStokesBrinkmann::solve_momentum()
{
    
    // xi = u + dt/beta * g



}