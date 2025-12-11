#include "NavierStokesBrinkmann.hpp"

int main() {
    // Example parameters
    int Nx = 100;          // Number of grid points in each direction
    double L = 1.0;     // Domain size
    double dt = 0.0001;   // Time step

    NavierStokesBrinkmann nsb(Nx, L, dt);
    nsb.initializeFields();

    double time = dt;
    int num_steps = 50;

    nsb.compute_error(0.0);

    for (int step = 0; step < num_steps; ++step) {
        
        nsb.compute_pressure_star(time);
        nsb.compute_g_rhs(time);
        nsb.solve_for_eta(time);
        nsb.solve_for_zeta(time);
        nsb.solve_for_velocity(time);
        nsb.compute_velocity_divergence(time);
        nsb.solve_for_psi(time);
        nsb.solve_for_phi(time);
        nsb.solve_for_Phi(time);

        //nsb.solve_pressure_iterative_adi();
        //nsb.correct_velocity();


        nsb.update_pressure(time);
        nsb.compute_error(time);

        time += dt;
    }

    return 0;
}