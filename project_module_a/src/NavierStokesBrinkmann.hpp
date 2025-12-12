#ifndef NAVIER_STOKES_BRINKMANN_HPP
#define NAVIER_STOKES_BRINKMANN_HPP
#include <iostream>
#include <vector>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


class NavierStokesBrinkmann {

    public:
        // Constructor
        NavierStokesBrinkmann(int N_, double L_, double dt_):
                N(N_), L(L_), dt(dt_) {
                dx = L / (N - 0.5);
                pressure.resize(N*N*N, 0.0f);
                velocity.resize(3, std::vector<float>(N*N*N, 0.0f)); // 3D velocity field
                pressure_star.resize(N*N*N, 0.0f);
                psi.resize(N*N*N, 0.0f);
                phi.resize(N*N*N, 0.0f);
                Phi.resize(N*N*N, 0.0f);
                rhs.resize(N*N*N, 0.0f);
                velocity_old.resize(3, std::vector<float>(N*N*N, 0.0f));
                xi.resize(3, std::vector<float>(N*N*N, 0.0f));
                zeta.resize(3, std::vector<float>(N*N*N, 0.0f));
                eta.resize(3, std::vector<float>(N*N*N, 0.0f));
                g_rhs.resize(3, std::vector<float>(N*N*N, 0.0f));
            }


        // DATA:

        class Permeability { //K
            public:
                Permeability() {};

            double value(double x, double y, double z) {
                // Example: constant permeability
                return 1.0;
            }

        };

        class Forcing_term { // f
            public:
                Forcing_term() {};

            double value(int comp, double x, double y, double z, double t) {
                double nu = 1.0; // Must match class viscosity
                double K = 1.0;   // Must match Permeability
                double pi = M_PI;
                
                // Frequencies
                // For Velocity: uses 2*pi so it vanishes at walls (Dirichlet)
                double Av = 2.0 * pi; 
                // For Pressure: uses 1*pi so derivatives (sin) vanish at walls (Neumann)
                double Ap = 1.0 * pi; 

                // Pre-compute spatial terms
                double sx_v = sin(Av * x);
                double cx_v = cos(Av * x);
                double sy_v = sin(Av * y);
                double cy_v = cos(Av * y);
                
                // Pressure spatial structure: cos(pi*x)*cos(pi*y)*cos(pi*z)
                // Gradient of pressure introduces -pi*sin(pi*x)...
                double grad_p = 0.0;
                
                // Velocity structure: u ~ sin(2pi*x)cos(2pi*y)
                double u_val = 0.0;
                double dt_u = 0.0; // Time derivative of velocity
                double lap_u = 0.0; // Laplacian of velocity

                // Time oscillation
                double sin_t = sin(t);
                double cos_t = cos(t);

                if (comp == 0) { // x-component equation
                    u_val = sx_v * cy_v * sin_t;
                    dt_u  = sx_v * cy_v * cos_t; 
                    // Laplacian of sin(Ax)*cos(Ay) -> -(Ax^2 + Ay^2) * u
                    lap_u = -(Av*Av + Av*Av) * u_val; 
                    // dp/dx
                    grad_p = -Ap * sin(Ap*x) * cos(Ap*y) * cos(Ap*z) * sin_t;
                }
                else if (comp == 1) { // y-component equation
                    u_val = -cx_v * sy_v * sin_t;
                    dt_u  = -cx_v * sy_v * cos_t;
                    lap_u = -(Av*Av + Av*Av) * u_val;
                    // dp/dy
                    grad_p = -Ap * cos(Ap*x) * sin(Ap*y) * cos(Ap*z) * sin_t;
                }
                else if (comp == 2) { // z-component equation
                    u_val = 0.0; 
                    dt_u  = 0.0;
                    lap_u = 0.0;
                    // dp/dz
                    grad_p = -Ap * cos(Ap*x) * cos(Ap*y) * sin(Ap*z) * sin_t;
                }

                // Solve for f:  du/dt = -dp/dx + nu*Lap(u) - (nu/K)*u + f
                // f = du/dt + dp/dx - nu*Lap(u) + (nu/K)*u
                return dt_u + grad_p - nu * lap_u + (nu / K) * u_val;
            }
        };

        class Exact_solution { // u_exact, p_exact
            public:
                Exact_solution() {};

            double value(int comp, double x, double y, double z, double t) {
                double pi = M_PI;
                double Av = 2.0 * pi; // Velocity freq
                double Ap = 1.0 * pi; // Pressure freq
                double sin_t = sin(t);

                if (comp == 0) { // u
                    return sin(Av * x) * cos(Av * y) * sin_t;
                }
                if (comp == 1) { // v
                    return -cos(Av * x) * sin(Av * y) * sin_t;
                }
                if (comp == 2) { // w
                    return 0.0;
                }
                if (comp == 3) { // pressure
                    // p = cos(pi*x)cos(pi*y)cos(pi*z)sin(t)
                    // Normal derivative at walls (e.g. x=0) involves sin(0) = 0.
                    // This satisfies Homogeneous Neumann.
                    return cos(Ap * x) * cos(Ap * y) * cos(Ap * z) * sin_t;
                }
                return 0.0;
            }
        };

    

        
        //HELPER METHODS:

        double compute_beta(double x, double y, double z);
        double compute_gamma(double x, double y, double z);

        double compute_pressure_star_gradient_x(int i, int j, int k);
        double compute_pressure_star_gradient_y(int i, int j, int k);
        double compute_pressure_star_gradient_z(int i, int j, int k);

        double compute_eta_xx(int comp, int i, int j, int k);
        double compute_zeta_yy(int comp, int i, int j, int k);
        double compute_vel_zz(int comp, int i, int j, int k);

        double delta_u(int comp, double x,double y,double z,double time);

        double Neumann_bc(int comp,double x, double y, double z, double time);


        //ALGORITHM METHODS:

        void initializeFields();

        void compute_pressure_star(double t);

        void compute_g_rhs(double t);

        void solve_for_eta(double time);
        void solve_for_zeta(double time);
        void solve_for_velocity(double time);

        void compute_velocity_divergence();
        void compute_velocity_divergence(double time);

        void solve_for_psi(double time);
        void solve_for_phi(double time);
        void solve_for_Phi(double time);
        void update_pressure(double t);

        void compute_error(double current_time);  
        void solve_pressure_iterative_adi(); 
        void correct_velocity();

        void get_error(double &total_error_vel, double &total_error_pres, double current_time);





    protected:

        //mesh parameters
        int N;
        double L;
        double dx;
        double dt;


        // DATA:

        const double viscosity = 1.0;
        Permeability permeability;
        Forcing_term forcing_term;
        Exact_solution exact_solution;


        // FIELDS:

        std::vector<float> pressure;
        std::vector<float> pressure_star;
        std::vector<float> psi;
        std::vector<float> phi;
        std::vector<float> Phi;
        std::vector<float> rhs;
        std::vector<std::vector<float> > velocity;
        std::vector<std::vector<float> > velocity_old;
        std::vector<std::vector<float> > xi;
        std::vector<std::vector<float> > zeta;
        std::vector<std::vector<float> > eta;
        std::vector<std::vector<float> > g_rhs;
        




};




#endif