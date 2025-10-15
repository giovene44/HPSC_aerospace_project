#include "navier_stokes_brinkmann.hpp"

void NavierStokesBrinkmann::assemble()


void NavierStokesBrinkmann::assemble_rhs()
{

    //this method assembles the g vector for the current time step.
    //(g vector is the rhs of the momentum equation!)

    double t = 0.0; // Placeholder for current time, should be set appropriately

    for (int i=0; i<g_rhs.size(); i++){
        for (int j=0; j<g_rhs[0].size(); j++)
        {
        g_rhs[i][j] = forcing_term.value(i,j,t) - gradient_pressure[i][j] 
        - eta*velocity_predictor[i][j]/(2.0*K[i][j])
        + eta/2.0 * ()
        }
    }

}