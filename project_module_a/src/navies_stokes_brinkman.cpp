#include "navier_stokes_brinkmann.hpp"
#include "VectorVariable.hpp"

void NavierStokesBrinkmann::assemble()

    void NavierStokesBrinkmann::assemble_g()
{

    // this method assembles the g vector for the current time step.
    //(g vector is the rhs of the momentum equation!)

    double t = 0.0; // Placeholder for current time, (TODO) should be set appropriately

    for (int i = 0; i < g_rhs.size(); i++)
    {                                             // i indicates the component among the 3 components of the vector
        for (int j = 0; j < g_rhs[0].size(); j++) // j indicates the index of the element in the vector
        {

            float eta_local_second_derivative = eta.second_derivative(i, 0, j);   // x direction
            float zeta_local_second_derivative = zeta.second_derivative(i, 1, j); // y direction
            float u_local_second_derivative = u.second_derivative(i, 2, j);       // z direction

            g_rhs[i][j] = forcing_term.value(i, j, t) - gradient_pressure[i][j] - eta * velocity_predictor[i][j] / (2.0 * /*K[i][j] or K.get(i,j)*/) + eta / 2.0 * (eta_local_second_derivative + zeta_local_second_derivative + u_local_second_derivative);
        }
    }
}

auto NavierStokesBrinkmann::assemble_rhs(auto output, auto vector1, auto vector2)
{
    for (int a = 0; a < 3; a++)
    { // a indicates the component among the 3 components of the vector
        for (int i = 0; i < Nx; i++)
        { // i indicates the index of the element in the vector
            for (int j = 0; j < Ny; j++)
            {
                for (int k = 0; k < Nz; k++)
                {
                    output.set(a, i, j, k) = vector1.value(a, i, j, k) - vector2.value(a, i, j, k);
                }
            }
        }
    }
}




void NavierStokesBrinkmann::solve_momentum()
{
    // set the righ t-hand side vector xi
    for (int a = 0; a < 3; a++)
    { // a indicates the component among the 3 components of the vector
        for (int i = 0; i < Nx; i++)
        { // i indicates the index of the element in the vector
            for (int j = 0; j < Ny; j++)
            {
                for (int k = 0; k < Nz; k++)
                {

                    xi.set(a, i, j, k) = u.value(a, i, j, k) + dt / beta(i, j, k) * g_rhs.value(a, i, j, k);
                }
            }
        }
    }
}