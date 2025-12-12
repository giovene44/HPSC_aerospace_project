#include "NavierStokesBrinkmann.hpp"

int main() {
    // Example parameters
    int Nx = 10;          // Number of grid points in each direction
    double L = 1.0;     // Domain size
    double dt = 0.0001;   // Time step
    int S=4;
    std::vector<double> vel_errors(S,0.0);
    std::vector<double> pres_errors(S,0.0); 

    for(int i = 0; i<S; i++){

        NavierStokesBrinkmann nsb(Nx, L, dt);
        nsb.initializeFields();

        double time = dt;
        int num_steps = 10;

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

        double a = 0.0;
        double b = 0.0;
        nsb.get_error(a,b,time);
        vel_errors[i]=a;
        pres_errors[i]=b;


        Nx *=2;


    }

    for (int i = 0; i < S - 1; ++i) {
        double vel_rate = std::log(vel_errors[i] / vel_errors[i + 1]) / std::log(2.0);
        double pres_rate = std::log(pres_errors[i] / pres_errors[i + 1]) / std::log(2.0);
        std::cout << "Refinement " << i << ": Velocity convergence rate = " << vel_rate 
                  << ", Pressure convergence rate = " << pres_rate << std::endl;
    }


    return 0;
}