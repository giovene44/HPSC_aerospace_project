#include "NavierStokesBrinkmann.hpp"

int Idx(int i, int j, int k, int N) {
    return i + j * N + k * N * N;
}

void tridiagonal_solver(const std::vector<float>& a, const std::vector<float>& b,
                        const std::vector<float>& c, std::vector<float>& d, int n) {
    std::vector<float> c_star(n, 0.0);
    std::vector<float> d_star(n, 0.0);

    c_star[0] = c[0] / b[0];
    d_star[0] = d[0] / b[0];

    for (int i = 1; i < n; ++i) {
        double m = b[i] - a[i] * c_star[i - 1];
        c_star[i] = c[i] / m;
        d_star[i] = (d[i] - a[i] * d_star[i - 1]) / m;
    }

    d[n - 1] = d_star[n - 1];
    for (int i = n - 2; i >= 0; --i) {
        d[i] = d_star[i] - c_star[i] * d[i + 1];
    }
}

double NavierStokesBrinkmann::delta_u(int comp, double x,double y,double z,double time){
    return exact_solution.value(comp,x,y,z,time)-exact_solution.value(comp,x,y,z,time-dt);
}

double NavierStokesBrinkmann::compute_beta(double x, double y, double z) {
    return 1.0 + dt * viscosity / (2.0 * permeability.value(x, y, z));
}

double NavierStokesBrinkmann::compute_gamma(double x, double y, double z) {
    return dt * viscosity / (2.0 * compute_beta(x, y, z));
}

void NavierStokesBrinkmann::initializeFields() {

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                double x = i * dx;
                double y = j * dx;
                double z = k * dx;

                int idx = i + j * N + k * N * N;

                // Initialize pressure field
                pressure[idx] = exact_solution.value(3, x, y, z, 0.0);
                pressure_star[idx] = pressure[idx];
                

                // Initialize velocity field
                
                velocity[0][idx] = exact_solution.value(0, x+dx/2, y, z, 0.0);
                velocity[1][idx] = exact_solution.value(1, x, y+dx/2, z, 0.0);
                velocity[2][idx] = exact_solution.value(2, x, y, z+dx/2, 0.0);

            }
        }
    }


    // initializa also eta and zeta
    for (int i = 0; i < N * N * N; ++i) {
        
        eta[0][i] = velocity[0][i];
        eta[1][i] = velocity[1][i];
        eta[2][i] = velocity[2][i];
        
        zeta[0][i] = velocity[0][i];
        zeta[1][i] = velocity[1][i];
        zeta[2][i] = velocity[2][i];
    }
    
}

double NavierStokesBrinkmann::compute_pressure_star_gradient_x(int i, int j, int k) {
    // Central difference for pressure gradient in x-direction
    // i am shifting to the velocity nodes!

    //check that i < N-1:
    if (i >= N - 1) {
        return 0.0;
    }

    int idx_plus = Idx(i + 1, j, k, N);
    int idx_minus = Idx(i, j, k, N);

    return (pressure_star[idx_plus] - pressure_star[idx_minus]) / dx;
}

double NavierStokesBrinkmann::compute_pressure_star_gradient_y(int i, int j, int k) {
    // Central difference for pressure gradient in y-direction
    // i am shifting to the velocity nodes!

    //check that j < N-1:
    if (j >= N - 1) {
        return 0.0;
    }

    int idx_plus = Idx(i, j + 1, k, N);
    int idx_minus = Idx(i, j, k, N);

    return (pressure_star[idx_plus] - pressure_star[idx_minus]) / dx;
}

double NavierStokesBrinkmann::compute_pressure_star_gradient_z(int i, int j, int k) {
    // Central difference for pressure gradient in z-direction
    // i am shifting to the velocity nodes!

    //check that k < N-1:
    if (k >= N - 1) {
        return 0.0;
    }

    int idx_plus = Idx(i, j, k + 1, N);
    int idx_minus = Idx(i, j, k, N);

    return (pressure_star[idx_plus] - pressure_star[idx_minus]) / dx;
}

double NavierStokesBrinkmann::compute_eta_xx(int comp, int i, int j, int k) {
    // Second derivative in x-direction for eta
    if (i < 0 || i > N - 1) {
        throw std::out_of_range("Index i out of range for eta second derivative computation");
    }

    if (i == 0 || i == N - 1) {
        //one-sided difference
        int idx = Idx(i, j, k, N);
        if (i == 0) {
            int idx_plus1 = Idx(i + 1, j, k, N);
            int idx_plus2 = Idx(i + 2, j, k, N);
            return (eta[comp][idx] - 2.0 * eta[comp][idx_plus1] + eta[comp][idx_plus2]) / (dx * dx);
        } 
        if (i == N - 1) {
            int idx_minus1 = Idx(i - 1, j, k, N);
            int idx_minus2 = Idx(i - 2, j, k, N);
            return (eta[comp][idx_minus2] - 2.0 * eta[comp][idx_minus1] + eta[comp][idx]) / (dx * dx);
        }
    }

    int idx_plus = Idx(i + 1, j, k, N);
    int idx_minus = Idx(i - 1, j, k, N);
    int idx = Idx(i, j, k, N);

    return (eta[comp][idx_plus] - 2.0 * eta[comp][idx] + eta[comp][idx_minus]) / (dx * dx);
}

double NavierStokesBrinkmann::compute_zeta_yy(int comp,int i, int j, int k) {
    // Second derivative in y-direction for zeta
    if (j < 0 || j > N - 1) {
        throw std::out_of_range("Index j out of range for zeta second derivative computation");
    }

    if (j == 0 || j == N - 1) {
        //one-sided difference
        int idx = Idx(i, j, k, N);
        if (j == 0) {
            int idx_plus1 = Idx(i, j + 1, k, N);
            int idx_plus2 = Idx(i, j + 2, k, N);
            return (zeta[comp][idx] - 2.0 * zeta[comp][idx_plus1] + zeta[comp][idx_plus2]) / (dx * dx);
        } 
        if (j == N - 1) {
            int idx_minus1 = Idx(i, j - 1, k, N);
            int idx_minus2 = Idx(i, j - 2, k, N);
            return (zeta[comp][idx_minus2] - 2.0 * zeta[comp][idx_minus1] + zeta[comp][idx]) / (dx * dx);
        }
    }

    int idx_plus = Idx(i, j + 1, k, N);
    int idx_minus = Idx(i, j - 1, k, N);
    int idx = Idx(i, j, k, N);

    return (zeta[comp][idx_plus] - 2.0 * zeta[comp][idx] + zeta[comp][idx_minus]) / (dx * dx);
}

double NavierStokesBrinkmann::compute_vel_zz(int comp, int i, int j, int k) {
    // Second derivative in z-direction for velocity
    if (k < 0 || k > N - 1) {
        throw std::out_of_range("Index k out of range for velocity second derivative computation");
    }

    if (k == 0 || k == N - 1) {
        //one-sided difference
        int idx = Idx(i, j, k, N);
        if (k == 0) {
            int idx_plus1 = Idx(i, j, k + 1, N);
            int idx_plus2 = Idx(i, j, k + 2, N);
            return (velocity[comp][idx] - 2.0 * velocity[comp][idx_plus1] + velocity[comp][idx_plus2]) / (dx * dx);
        } 
        if (k == N - 1) {
            int idx_minus1 = Idx(i, j, k - 1, N);
            int idx_minus2 = Idx(i, j, k - 2, N);
            return (velocity[comp][idx_minus2] - 2.0 * velocity[comp][idx_minus1] + velocity[comp][idx]) / (dx * dx);
        }
    }

    int idx_plus = Idx(i, j, k + 1, N);
    int idx_minus = Idx(i, j, k - 1, N);
    int idx = Idx(i, j, k, N);

    return (velocity[comp][idx_plus] - 2.0 * velocity[comp][idx] + velocity[comp][idx_minus]) / (dx * dx);
}

void NavierStokesBrinkmann::compute_g_rhs(double t) {
    // Compute the right-hand side for the g equation

    double t_mid = t - dt / 2.0;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {

                int idx = Idx(i, j, k, N);

                double x = i * dx;
                double y = j * dx;
                double z = k * dx;
 
                double g_rhs_x = forcing_term.value(0, x+dx/2, y, z,t_mid);
                double g_rhs_y = forcing_term.value(1, x, y+dx/2, z,t_mid); 
                double g_rhs_z = forcing_term.value(2, x, y, z+dx/2,t_mid);

                g_rhs_x -= compute_pressure_star_gradient_x(i, j, k);
                g_rhs_y -= compute_pressure_star_gradient_y(i, j, k);
                g_rhs_z -= compute_pressure_star_gradient_z(i, j, k);

                double temp = viscosity / permeability.value(x, y, z) / 2.0;

                g_rhs_x -= temp * velocity[0][idx];
                g_rhs_y -= temp * velocity[1][idx];
                g_rhs_z -= temp * velocity[2][idx];

                g_rhs_x += viscosity/2.0 * (compute_eta_xx(0, i, j, k) + compute_zeta_yy(0, i, j, k) + compute_vel_zz(0, i, j, k));
                g_rhs_y += viscosity/2.0 * (compute_eta_xx(1, i, j, k) + compute_zeta_yy(1, i, j, k) + compute_vel_zz(1, i, j, k));
                g_rhs_z += viscosity/2.0 * (compute_eta_xx(2, i, j, k) + compute_zeta_yy(2, i, j, k) + compute_vel_zz(2, i, j, k));

                //recycle g_rhs to store the rhs for eta
                g_rhs[0][idx] = dt/compute_beta(x+dx/2, y, z)*g_rhs_x + velocity[0][idx];
                g_rhs[1][idx] = dt/compute_beta(x, y+dx/2, z)*g_rhs_y + velocity[1][idx];
                g_rhs[2][idx] = dt/compute_beta(x, y, z+dx/2)*g_rhs_z + velocity[2][idx];

                g_rhs[0][idx] -= eta[0][idx];
                g_rhs[1][idx] -= eta[1][idx];
                g_rhs[2][idx] -= eta[2][idx];

            }
        }
    }
}

void NavierStokesBrinkmann::compute_pressure_star(double t) {
    // Update pressure_star field
    for (int i = 0; i < N * N * N; ++i) {
        pressure_star[i] = pressure[i] + Phi[i];
    }
}

void NavierStokesBrinkmann::solve_for_eta(double time) {

    // X-SWEEP!
    
    std::vector<float> a(N, 0.0);
    std::vector<float> b(N, 0.0);
    std::vector<float> c(N, 0.0);
    std::vector<float> d(N, 0.0);
    std::vector<float> solution(N, 0.0);

    // (1 - gamma*Dxx)(eta^(n+1)-eta^n) = g_rhs
    // Solve along x-direction for each (j,k)

    //handle knwon lines:

    //face k=0:
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int k = 0; // known wall at k=0
            int idx = Idx(i, j, k, N);
            eta[0][idx] += delta_u(0, i*dx+dx/2, j*dx, 0.0, time); 
            eta[1][idx] += delta_u(1, i*dx, j*dx+dx/2, 0.0, time);
        }
    }

    //face k=N-1:
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int k = N - 1; // known wall at k=N-1
            int idx = Idx(i, j, k, N);
            eta[2][idx] += delta_u(2, i*dx, j*dx, L, time); 
        }
    }

    //face j=0:
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            int j = 0; // known wall at j=0
            int idx = Idx(i, j, k, N);
            if(k!=0)eta[0][idx] += delta_u(1, i*dx+dx/2, 0.0, k*dx, time); 
            if(k!=(N-1))eta[2][idx] += delta_u(2, i*dx, 0.0, k*dx+dx/2, time);
        }
    }

    //face j=N-1:
    for (int i = 0; i < N; ++i) {
        for (int k = 1; k < N; ++k) {
            int j = N - 1; // known wall at j=N-1
            int idx = Idx(i, j, k, N);
            eta[1][idx] += delta_u(1, i*dx, L, k*dx+dx/2, time); 
        }
    }


    //eta_x: walls at j=0, k=0 are known (Dirichlet BCs)
    for (int j = 1; j < N; ++j) {
        for (int k = 1; k < N; ++k) {

            double y = j * dx;
            double z = k * dx;

            for (int i = 0; i < N; ++i) {
                double x = i * dx;

                double gamma = compute_gamma(x+dx/2, y, z);

                a[i] = -gamma / (dx * dx);
                b[i] = 1.0 + 2.0 * gamma / (dx * dx);
                c[i] = -gamma / (dx * dx);

                int idx = Idx(i, j, k, N);
                d[i] = g_rhs[0][idx]; // for eta_x
            }


            //taylor technique:
            //b[0]-=2.0*a[0];
            //c[0]+=1.0/3.0*a[0];
            //d[0]-=8.0/3.0*a[0]*delta_u(0,0,y,z,time);

            b[N - 1] = 1.0;
            a[N - 1] = 0.0;

            d[N-1]=exact_solution.value(0, L, j*dx, k*dx, time)
                    -exact_solution.value(0,L, j*dx, k*dx,time-dt);

            
            //use divergence trick: 
            b[0] = 1.0;
            c[0] = 0.0;
            double dvdy = delta_u(1,0,y+dx/2,z,time)-delta_u(1,0,y-dx/2,z,time);                 
            dvdy /= dx;

            double dwdz = delta_u(2,0,y,z+dx/2,time)-delta_u(2,0,y,z-dx/2,time);
            dwdz /= dx;

            //divergence = dudx + dvdy + dwdz =0  =>  dudx = - (dvdy + dwdz).
            double dudx = -dvdy - dwdz;

            d[0]= delta_u(0,0,y,z,time) + dudx*dx/2.0; // Dirichlet BC at the start
            

            tridiagonal_solver(a, b, c, d, N);
            

            // eta_x update by adding delta eta to old eta
            for (int i = 0; i < N; ++i) {
                eta[0][Idx(i, j, k, N)] += d[i];
            }
        }


    }

    //eta_y: walls at j=N-1, k=0 are known (Dirichlet BCs)
    for (int j = 0; j < N - 1; ++j) {
        for (int k = 1; k < N; ++k) {

            double y = j * dx;
            double z = k * dx;

            for (int i = 0; i < N; ++i) {
                double x = i * dx;

                double gamma = compute_gamma(x, y+dx/2, z);

                a[i] = -gamma / (dx * dx);
                b[i] = 1.0 + 2.0 * gamma / (dx * dx);
                c[i] = -gamma / (dx * dx);

                int idx = Idx(i, j, k, N);
                d[i] = g_rhs[1][idx]; // for eta_y
            }

            // Apply BCs: 
            b[0] = 1.0;
            c[0] = 0.0;

            d[0] = delta_u(1,0,y+dx/2,z,time);

            //ghost node on right wall j=N-1
        
            b[N - 1] -= c[N -1]; 
            d[N -1] -= 2.0*c[N -1]*delta_u(1,L,y+dx/2,z,time);

            tridiagonal_solver(a, b, c, d, N);

            // eta_y update by adding delta eta to old eta
            for (int i = 0; i < N; ++i) {
                eta[1][Idx(i, j, k, N)] += d[i];
            }
        }
    }

    //eta_z: walls at j=0, k=N-1 are known (Dirichlet BCs)
    for( int j = 1; j < N; ++j) {
        for (int k = 0; k < N - 1; ++k) {

            double y = j * dx;
            double z = k * dx;

            for (int i = 0; i < N; ++i) {
                double x = i * dx;

                double gamma = compute_gamma(x, y, z+dx/2);

                a[i] = -gamma / (dx * dx);
                b[i] = 1.0 + 2.0 * gamma / (dx * dx);
                c[i] = -gamma / (dx * dx);

                int idx = Idx(i, j, k, N);
                d[i] = g_rhs[2][idx]; // for eta_z
            }

            // Apply BCs:
            b[0] = 1.0;
            c[0] = 0.0;

            d[0] = delta_u(2,0,y,z+dx/2,time);

            //ghost node on right wall k=N-1
        
            b[N - 1] -= c[N -1]; 
            d[N -1] -= 2.0*c[N -1]*delta_u(2,L,y,z+dx/2,time);

            tridiagonal_solver(a, b, c, d, N);

            // eta_z update by adding delta eta to old eta
            for (int i = 0; i < N; ++i) {
                eta[2][Idx(i, j, k, N)] += d[i];
            }
        }
    }
}

void NavierStokesBrinkmann::solve_for_zeta(double time) {

    //prepare the rhs:

    for (int idx=0; idx<N*N*N; idx++){
        eta[0][idx]-=zeta[0][idx];
        eta[1][idx]-=zeta[1][idx];
        eta[2][idx]-=zeta[2][idx];
    }


    // Y-SWEEP!
    // Here we use the just computed eta as our rhs!
    
    std::vector<float> a(N, 0.0);
    std::vector<float> b(N, 0.0);
    std::vector<float> c(N, 0.0);
    std::vector<float> d(N, 0.0);
    std::vector<float> solution(N, 0.0);

    // (1 - gamma*Dyy)(zeta^(n+1)-zeta^n) = g_rhs
    // Solve along y-direction for each (i,k)

    //handle knwon lines:

    //face k=0:
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int k = 0; // known wall at k=0
            int idx = Idx(i, j, k, N);
            zeta[0][idx] += delta_u(0, i*dx+dx/2, j*dx, 0.0, time); 
            zeta[1][idx] += delta_u(1, i*dx, j*dx+dx/2, 0.0, time);
        }
    }

    //face k=N-1:
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int k = N - 1; // known wall at k=N-1
            int idx = Idx(i, j, k, N);
            zeta[2][idx] += delta_u(2, i*dx, j*dx, L, time); 
        }
    }

    //face i=0:
    for (int j = 0; j < N; ++j) {
        for (int k = 0; k < N; ++k) {
            int i = 0; // known wall at i=0
            int idx = Idx(i, j, k, N);
            if(k!=0)zeta[1][idx] += delta_u(0, 0.0, j*dx+dx/2, k*dx, time); 
            if(k!=N-1)zeta[2][idx] += delta_u(2, 0.0, j*dx, k*dx+dx/2, time);
        }
    }

    //face i=N-1:
    for (int j = 0; j < N; ++j) {
        for (int k = 1; k < N; ++k) {
            int i = N - 1; // known wall at i=N-1
            int idx = Idx(i, j, k, N);
            zeta[0][idx] += delta_u(0, L, j*dx, k*dx, time); 
        }
    }


    //zeta_x: walls at i=N-1, k=0 are known (Dirichlet BCs)
    for (int i = 0; i < N - 1; ++i) {
        for (int k = 1; k < N; ++k) {

            double x = i * dx;
            double z = k * dx;

            for (int j = 0; j < N; ++j) {
                double y = j * dx;

                double gamma = compute_gamma(x+dx/2, y, z);

                a[j] = -gamma / (dx * dx);
                b[j] = 1.0 + 2.0 * gamma / (dx * dx);
                c[j] = -gamma / (dx * dx); 

                int idx = Idx(i, j, k, N);
                d[j] = eta[0][idx]; // for zeta_x
            
            }

            // Apply BCs:
            b[0] = 1.0;
            c[0] = 0.0;

            d[0] = delta_u(0, x+dx/2, 0.0, z, time);

            //ghost node on wall j=N-1:

            b[N - 1] -= c[N -1];
            d[N -1] -= 2.0*c[N -1]*delta_u(0,x+dx/2,L,z,time);

            tridiagonal_solver(a, b, c, d, N);

            // zeta_x update by adding delta zeta to old zeta
            for (int j = 0; j < N; ++j) {
                zeta[0][Idx(i, j, k, N)] += d[j];
            }
        }               
    }

    //zeta_y: walls at i=0, k=0 are known (Dirichlet BCs)
    for (int i = 1; i < N; ++i) {
        for (int k = 1; k < N; ++k) {
            double x = i * dx;
            double z = k * dx;

            for (int j = 0; j < N; ++j) {
                double y = j * dx;

                double gamma = compute_gamma(x, y+dx/2, z);

                a[j] = -gamma / (dx * dx);
                b[j] = 1.0 + 2.0 * gamma / (dx * dx);
                c[j] = -gamma / (dx * dx); 

                int idx = Idx(i, j, k, N);
                d[j] = eta[1][idx]; // for zeta_y
            }

            //Taylor technique:
            //b[0]-=2.0*a[0];
            //c[0]+=1.0/3.0*a[0];
            //d[0]-=8.0/3.0*a[0]*delta_u(1,x,0,z,time);


            // Apply BCs:
            //use divergence trick at j=0
            

            b[0] = 1.0;
            c[0] = 0.0;
            double dudx = delta_u(0,x+dx/2,0.0,z,time)-delta_u(0,x-dx/2,0.0,z,time);
            dudx /= dx;
            double dwdz = delta_u(2,x,0.0,z+dx/2,time)-delta_u(2,x,0.0,z-dx/2,time);
            dwdz /= dx;
            double dvdy = -dudx - dwdz;
            d[0] = delta_u(1,x,0.0,z,time) + dvdy*dx/2.0; // Dirichlet BC at the start 
            
            

            //simple Dirichlet at j=N-1
            b[N - 1] = 1.0;
            a[N - 1] = 0.0;
            d[N -1] = delta_u(1,x,L,z,time);

            //solve linear system
            tridiagonal_solver(a, b, c, d, N);

            // zeta_y update by adding delta zeta to old zeta
            for (int j = 0; j < N; ++j) {
                zeta[1][Idx(i, j, k, N)] += d[j];
            }
        }
    }

    //zeta_z: walls at i=0, k=N-1 are known (Dirichlet BCs)
    for (int i = 1; i < N; ++i) {
        for (int k = 0; k < N - 1; ++k) {
            double x = i * dx;
            double z = k * dx;

            for (int j = 0; j < N; ++j) {
                double y = j * dx;

                double gamma = compute_gamma(x, y, z+dx/2);

                a[j] = -gamma / (dx * dx);
                b[j] = 1.0 + 2.0 * gamma / (dx * dx);
                c[j] = -gamma / (dx * dx); 

                int idx = Idx(i, j, k, N);
                d[j] = eta[2][idx]; // for zeta_z
            }

            // Apply BCs:
            b[0] = 1.0;
            c[0] = 0.0;

            d[0] = delta_u(2,x,0.0,z+dx/2,time);

            //ghost node on wall k=N-1:
            b[N - 1] -= c[N -1];
            d[N -1] -= 2.0*c[N -1]*delta_u(2,x,L,z+dx/2,time);

            tridiagonal_solver(a, b, c, d, N);

            // zeta_z update by adding delta zeta to old zeta
            for (int j = 0; j < N; ++j) {
                zeta[2][Idx(i, j, k, N)] += d[j];
            }
        }
    }
}

void NavierStokesBrinkmann::solve_for_velocity(double time){


    //prepare the rhs:

    for (int idx=0; idx<N*N*N; idx++){
        zeta[0][idx]-=velocity[0][idx];
        zeta[1][idx]-=velocity[1][idx];
        zeta[2][idx]-=velocity[2][idx];
    }

    //Z-SWEEP!
    // Here we use the just computed zeta as our rhs!

    std::vector<float> a(N, 0.0);
    std::vector<float> b(N, 0.0);
    std::vector<float> c(N, 0.0);
    std::vector<float> d(N, 0.0);
    std::vector<float> solution(N, 0.0);

    // (1 - gamma*Dzz)(u^(n+1)-u^n) = zeta
    // Solve along z-direction for each (i,j)

    //handle knwon lines:

    //face j=0:
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            int j = 0; // known wall at j=0
            int idx = Idx(i, j, k, N);
            velocity[0][idx] += delta_u(0, i*dx+dx/2, 0.0, k*dx, time); 
            velocity[2][idx] += delta_u(2, i*dx, 0.0, k*dx+dx/2, time);
        }
    }

    //face j=N-1:
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            int j = N - 1; // known wall at j=N-1
            int idx = Idx(i, j, k, N);
            velocity[1][idx] += delta_u(1, i*dx, L, k*dx, time); 
        }
    }

    //face i=0:
    for (int j = 0; j < N; ++j) {
        for (int k = 0; k < N; ++k) {
            int i = 0; // known wall at i=0
            int idx = Idx(i, j, k, N);
            if(j!=N-1)velocity[1][idx] += delta_u(0, 0.0, j*dx+dx/2, k*dx, time); 
            if(j!=0)velocity[2][idx] += delta_u(2, 0.0, j*dx, k*dx+dx/2, time);
        }
    }

    //face i = N-1:
    for (int j = 1; j < N; ++j) {
        for (int k = 0; k < N; ++k) {
            int i = N - 1; // known wall at i=N-1
            int idx = Idx(i, j, k, N);
            velocity[0][idx] += delta_u(0, L, j*dx, k*dx, time); 
        }
    }

    //velocity_x: walls at i=N-1, j=0 are known (Dirichlet BCs)
    for (int i = 0; i < N - 1; ++i) {
        for (int j = 1; j < N; ++j) {

            double x = i * dx;
            double y = j * dx;

            for (int k = 0; k < N; ++k) {
                double z = k * dx;

                double gamma = compute_gamma(x+dx/2, y, z);

                a[k] = -gamma / (dx * dx);
                b[k] = 1.0 + 2.0 * gamma / (dx * dx);
                c[k] = -gamma / (dx * dx); 

                int idx = Idx(i, j, k, N);
                d[k] = zeta[0][idx]; // for velocity_x
            }

            // Apply BCs:
            b[0] = 1.0;
            c[0] = 0.0;

            d[0] = delta_u(0, x+dx/2, y, 0.0, time);

            //ghost node on wall k=N-1:

            b[N - 1] -= c[N -1];
            d[N -1] -= 2.0*c[N -1]*delta_u(0,x+dx/2,y,L,time);

            tridiagonal_solver(a, b, c, d, N);

            // velocity_x update by adding delta velocity to old velocity
            for (int k = 0; k < N; ++k) {
                velocity[0][Idx(i, j, k, N)] += d[k];
            }
        }               
    }

    //velocity_y: walls at i=0, j=N-1 are known (Dirichlet BCs)
    for (int i = 1; i < N; ++i) {
        for (int j = 0; j < N - 1; ++j) {
            double x = i * dx;
            double y = j * dx;

            for (int k = 0; k < N; ++k) {
                double z = k * dx;

                double gamma = compute_gamma(x, y+dx/2, z);

                a[k] = -gamma / (dx * dx);
                b[k] = 1.0 + 2.0 * gamma / (dx * dx);
                c[k] = -gamma / (dx * dx); 

                int idx = Idx(i, j, k, N);
                d[k] = zeta[1][idx]; // for velocity_y
            }

            // Apply BCs:
            b[0] = 1.0;
            c[0] = 0.0;

            d[0] = delta_u(1, x, y+dx/2, 0.0, time);

            //ghost node on wall k=N-1:
            b[N - 1] -= c[N -1];
            d[N -1] -= 2.0*c[N -1]*delta_u(1,x,y+dx/2,L,time);

            tridiagonal_solver(a, b, c, d, N);

            // velocity_y update by adding delta velocity to old velocity
            for (int k = 0; k < N; ++k) {
                velocity[1][Idx(i, j, k, N)] += d[k];
            }
        }
    }

    //velocity_z: walls at i=0, j=0 are known (Dirichlet BCs)
    for (int i = 1; i < N; ++i) {
        for (int j = 1; j < N; ++j) {
            double x = i * dx;
            double y = j * dx;

            for (int k = 0; k < N; ++k) {
                double z = k * dx;

                double gamma = compute_gamma(x, y, z+dx/2);

                a[k] = -gamma / (dx * dx);
                b[k] = 1.0 + 2.0 * gamma / (dx * dx);
                c[k] = -gamma / (dx * dx); 

                int idx = Idx(i, j, k, N);
                d[k] = zeta[2][idx]; // for velocity_z
            }

            // Apply BCs:


            //Taylor technique:
            //b[0]-=2.0*a[0];
            //c[0]+=1.0/3.0*a[0];
            //d[0]-=8.0/3.0*a[0]*delta_u(2,x,y,0,time);

            
            //use divergence trick at k=0
            b[0] = 1.0;
            c[0] = 0.0;
            double dudx = delta_u(0,x+dx/2,y,0.0,time)-delta_u(0,x-dx/2,y,0.0,time);
            dudx /= dx;
            double dvdy = delta_u(1,x,y+dx/2,0.0,time)-delta_u(1,x,y-dx/2,0.0,time);
            dvdy /= dx;
            double dwdz = -dudx - dvdy;
            d[0] = delta_u(2,x,y,0.0,time) + dwdz*dx/2.0; // Dirichlet BC at the start
            

            //simple Dirichlet at k=N-1
            b[N - 1] = 1.0;
            a[N - 1] = 0.0;
            d[N -1] = delta_u(2,x,y,L,time);

            //solve linear system
            tridiagonal_solver(a, b, c, d, N);

            // velocity_z update by adding delta velocity to old velocity
            for (int k = 0; k < N; ++k) {
                velocity[2][Idx(i, j, k, N)] += d[k];
            }
        }               
    }
}

void NavierStokesBrinkmann::compute_velocity_divergence(){

    //we need to shift to the pressure nodes!
    //i can skip the boundaries i=0, j=0, k=0 cause their corresponding rhs value will be replaced by BCs

    for(int i=0;i<N;++i){
        for(int j=0;j<N;++j){
            for(int k=0;k<N;++k){

                int idx = Idx(i,j,k,N);

                double dudx, dvdy, dwdz;
                if(i==0 || j==0 || k==0)
                    rhs[idx]=0.0; //will be replaced by BCs
                else{
                    dudx = (velocity[0][Idx(i,j,k,N)] - velocity[0][Idx(i-1,j,k,N)])/dx;
                    dvdy = (velocity[1][Idx(i,j,k,N)] - velocity[1][Idx(i,j-1,k,N)])/dx;
                    dwdz = (velocity[2][Idx(i,j,k,N)] - velocity[2][Idx(i,j,k-1,N)])/dx;

                    rhs[idx] = -(dudx + dvdy + dwdz)/dt;
                }
            }
        }
    }
}

double NavierStokesBrinkmann::Neumann_bc(int comp,double x, double y, double z, double time){
    return 0.0; //homogeneous Neumann BC
}

void NavierStokesBrinkmann::solve_for_psi(double time){

    //(I - Dxx)psi = rhs

    std::vector<float> a(N, 0.0);
    std::vector<float> b(N, 0.0);
    std::vector<float> c(N, 0.0);
    std::vector<float> d(N, 0.0);

    // Solve along x-direction for each (j,k)
    for (int j = 0; j < N; ++j) {
        for (int k = 0; k < N; ++k) {

            double y= j * dx;
            double z= k * dx;

            for (int i = 0; i < N; ++i) {
                a[i] = -1.0 / (dx * dx);
                b[i] = 1.0 + 2.0 / (dx * dx);
                c[i] = -1.0 / (dx * dx);

                int idx = Idx(i, j, k, N);
                d[i] = rhs[idx];
            }

            // Apply BCs:

            c[0] += a[0]; // Neumann BC at the start
            d[0] += 2.0 * a[0] * dx * Neumann_bc(0,0,y,z,time);

            b[N - 1] += c[N -1]; // Neumann BC at the end
            d[N -1] -= c[N -1] * dx * Neumann_bc(0,L,y,z,time);


            tridiagonal_solver(a, b, c, d, N);

            // psi update
            for (int i = 0; i < N; ++i) {
                int idx = Idx(i, j, k, N);
                psi[idx] = d[i];
            }
        }
    }
}

void NavierStokesBrinkmann::solve_for_phi(double time){
    //(I - Dyy)phi = psi

    std::vector<float> a(N, 0.0);
    std::vector<float> b(N, 0.0);
    std::vector<float> c(N, 0.0);
    std::vector<float> d(N, 0.0);

    // Solve along y-direction for each (i,k)
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {

            double x= i * dx;
            double z= k * dx;

            for (int j = 0; j < N; ++j) {
                a[j] = -1.0 / (dx * dx);
                b[j] = 1.0 + 2.0 / (dx * dx);
                c[j] = -1.0 / (dx * dx);

                int idx = Idx(i, j, k, N);
                d[j] = psi[idx];
            }

            // Apply BCs:

            c[0] += a[0]; // Neumann BC at the start
            d[0] += 2.0 * a[0] * dx * Neumann_bc(1,x,0.0,z,time);

            b[N - 1] += c[N -1]; // Neumann BC at the end
            d[N -1] -= c[N -1] * dx * Neumann_bc(1,x,L,z,time);

            tridiagonal_solver(a, b, c, d, N);

            // phi update
            for (int j = 0; j < N; ++j) {
                int idx = Idx(i, j, k, N);
                phi[idx] = d[j];
            }
        }
    }    
}

void NavierStokesBrinkmann::solve_for_Phi(double time){
    //(I - Dzz)Phi = phi

    std::vector<float> a(N, 0.0);
    std::vector<float> b(N, 0.0);
    std::vector<float> c(N, 0.0);
    std::vector<float> d(N, 0.0);

    // Solve along z-direction for each (i,j)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {

            double x= i * dx;
            double y= j * dx;

            for (int k = 0; k < N; ++k) {
                a[k] = -1.0 / (dx * dx);
                b[k] = 1.0 + 2.0 / (dx * dx);
                c[k] = -1.0 / (dx * dx);

                int idx = Idx(i, j, k, N);
                d[k] = phi[idx];
            }

            // Apply BCs:

            c[0] += a[0]; // Neumann BC at the start
            d[0] += 2.0 * a[0] * dx * Neumann_bc(2,x,y,0.0,time);

            b[N - 1] += c[N -1]; // Neumann BC at the end
            d[N -1] -= c[N -1] * dx * Neumann_bc(2,x,y,L,time);

            tridiagonal_solver(a, b, c, d, N);

            // Phi update
            for (int k = 0; k < N; ++k) {
                int idx = Idx(i, j, k, N);
                Phi[idx] = d[k];
            }
        }
    }    
}

void NavierStokesBrinkmann::update_pressure(double t) {
    // Update pressure field
    for (int i = 0; i < N * N * N; ++i) {
        pressure[i] += Phi[i];
        pressure[i] -= pressure[0];
        pressure[i] += exact_solution.value(3, 0.0, 0.0, 0.0, t);
    }
}

void NavierStokesBrinkmann::compute_error(double current_time) {

    // Compute error norms between numerical and exact solutions
    double error_u = 0.0;
    double error_v = 0.0;
    double error_w = 0.0;
    double error_p = 0.0;
    double vel_norm = 0.0;
    double pres_norm = 0.0;
    double vel_norm_comparison = 0.0;
    double pres_norm_comparison = 0.0;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                double x = i * dx;
                double y = j * dx;
                double z = k * dx;

                int idx = Idx(i, j, k, N);

                double u_exact = exact_solution.value(0, x+dx/2, y, z, current_time);
                double v_exact = exact_solution.value(1, x, y+dx/2, z, current_time);
                double w_exact = exact_solution.value(2, x, y, z+dx/2, current_time);
                double p_exact = exact_solution.value(3, x, y, z, current_time);

                error_u += (velocity[0][idx] - u_exact) * (velocity[0][idx] - u_exact);
                error_v += (velocity[1][idx] - v_exact) * (velocity[1][idx] - v_exact);
                error_w += (velocity[2][idx] - w_exact) * (velocity[2][idx] - w_exact);
                error_p += (pressure[idx] - p_exact) * (pressure[idx] - p_exact);

                double u_exact1 = exact_solution.value(0, x+dx/2, y, z, 1.0);
                double v_exact1 = exact_solution.value(1, x, y+dx/2, z, 1.0);
                double w_exact1 = exact_solution.value(2, x, y, z+dx/2, 1.0);
                double p_exact1 = exact_solution.value(3, x, y, z, 1.0);


                

                vel_norm_comparison += u_exact1 * u_exact1 + v_exact1 * v_exact1 + w_exact1 * w_exact1;
                pres_norm_comparison += p_exact1 * p_exact1;
                vel_norm += u_exact * u_exact + v_exact * v_exact + w_exact * w_exact;
                pres_norm += p_exact * p_exact;
            }
        }
    }

    double dV = dx * dx * dx;

    double total_error_vel = std::sqrt((error_u + error_v + error_w)*dV);
    double total_error_pres = std::sqrt(error_p * dV);
    double relative_error_vel = total_error_vel / std::sqrt(vel_norm_comparison * dV);
    double relative_error_pres = total_error_pres / std::sqrt(pres_norm_comparison * dV);


    std::cout << "----------------------------------------" << std::endl;
    std::cout << "At time: " << current_time << std::endl;
    std::cout << "Velocity L2 Error: " << total_error_vel << ", Relative Error: " << relative_error_vel << std::endl;
    std::cout << "Pressure L2 Error: " << total_error_pres << ", Relative Error: " << relative_error_pres << std::endl;
}


void NavierStokesBrinkmann::solve_pressure_iterative_adi() {
    // Parametri
    int max_iter = 100;      
    double tol = 1e-6;       
    double dtau = dx * dx;   // Pseudo-time step
    double alpha = dtau;     // Fattore moltiplicativo (dtau o dtau/2)

    double alpha_dx2 = alpha / (dx * dx);

    // Vettori Tridiagonali
    // Operatore: (I - alpha * D2) delta_p = ...
    // Diagonale (b): 1 + 2 * alpha/dx^2
    // Extra-diag (a, c): -alpha/dx^2
    std::vector<float> a(N, -alpha_dx2);
    std::vector<float> b(N, 1.0 + 2.0 * alpha_dx2);
    std::vector<float> c(N, -alpha_dx2);
    std::vector<float> d(N);
    
    // Vettori di supporto per ripristinare le BC
    float b0_orig = b[0];
    float bN_orig = b[N-1];

    for (int iter = 0; iter < max_iter; ++iter) {
        double max_resid = 0.0;

        // --- 1. Calcolo Residuo ---
        // Poisson: Laplaciano(p) = -rhs
        // Residuo = (-rhs) - Laplaciano(p)
        
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                for (int k = 0; k < N; ++k) {
                    int idx = Idx(i, j, k, N);

                    // Gestione Neumann per il calcolo del Laplaciano attuale (p)
                    // Se siamo al bordo, p_vicino = p_corrente (derivata nulla)
                    double p_c = pressure[idx];
                    double p_im = (i == 0)   ? p_c : pressure[Idx(i-1, j, k, N)];
                    double p_ip = (i == N-1) ? p_c : pressure[Idx(i+1, j, k, N)];
                    double p_jm = (j == 0)   ? p_c : pressure[Idx(i, j-1, k, N)];
                    double p_jp = (j == N-1) ? p_c : pressure[Idx(i, j+1, k, N)];
                    double p_km = (k == 0)   ? p_c : pressure[Idx(i, j, k-1, N)];
                    double p_kp = (k == N-1) ? p_c : pressure[Idx(i, j, k+1, N)];

                    double lap_p = (p_im + p_ip + p_jm + p_jp + p_km + p_kp - 6.0*p_c) / (dx*dx);

                    // CORREZIONE SEGNO: -rhs - lap_p
                    double r = -rhs[idx] - lap_p;
                    psi[idx] = r; // Salviamo il residuo in psi

                    if (std::abs(r) > max_resid) max_resid = std::abs(r);
                }
            }
        }

        if (max_resid < tol) break;

        // --- 2. X-Sweep ---
        // Risolviamo per phi (temp)
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                for (int i = 0; i < N; ++i) d[i] = alpha * psi[Idx(i, j, k, N)];

                // BC Neumann su Delta P: b[0] += a[0] (perché a[0] è negativo, riduce la diagonale)
                // In pratica: (1 + alpha/dx^2) p0 - alpha/dx^2 p1 = ...
                b[0] += a[0]; 
                b[N-1] += c[N-1];

                tridiagonal_solver(a, b, c, d, N);

                // Ripristino
                b[0] = b0_orig; 
                b[N-1] = bN_orig;

                for (int i = 0; i < N; ++i) phi[Idx(i, j, k, N)] = d[i];
            }
        }

        // --- 3. Y-Sweep ---
        // Risolviamo per Phi (temp) usando phi come rhs
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < N; ++k) {
                for (int j = 0; j < N; ++j) d[j] = phi[Idx(i, j, k, N)];

                b[0] += a[0]; 
                b[N-1] += c[N-1];

                tridiagonal_solver(a, b, c, d, N);

                b[0] = b0_orig; 
                b[N-1] = bN_orig;

                for (int j = 0; j < N; ++j) Phi[Idx(i, j, k, N)] = d[j];
            }
        }

        // --- 4. Z-Sweep ---
        // Risolviamo per Delta P e aggiorniamo Pressione
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                for (int k = 0; k < N; ++k) d[k] = Phi[Idx(i, j, k, N)];

                b[0] += a[0]; 
                b[N-1] += c[N-1];

                tridiagonal_solver(a, b, c, d, N);

                b[0] = b0_orig; 
                b[N-1] = bN_orig;

                // Aggiornamento Pressione
                for (int k = 0; k < N; ++k) {
                    pressure[Idx(i, j, k, N)] = d[k];
                }
            }
        }
    }
    
    // Ancoraggio della pressione (media zero o punto fisso) per evitare deriva
    double p_avg = 0.0;
    for(double val : pressure) p_avg += val;
    p_avg /= (N*N*N);
    // Correggi solo se necessario, o ancora meglio fissa p[0] al valore esatto se noto
    // per ora lasciamo fluttuare o sottraiamo la media
    for(int idx=0; idx<N*N*N; ++idx) pressure[idx] -= p_avg;
}

void NavierStokesBrinkmann::correct_velocity() {
    // Update velocity: u = u* - dt * grad(p)
    
    // 1. Correct U (X-component)
    // velocity[0] vive a i+1/2. Usa p[i+1] - p[i]
    for (int i = 0; i < N - 1; ++i) { // Nota il limite N-1
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                int idx = Idx(i, j, k, N);
                int idx_next = Idx(i + 1, j, k, N);
                
                double dp_dx = (pressure[idx_next] - pressure[idx]) / dx;
                velocity[0][idx] -= dt * dp_dx;
            }
        }
    }

    // 2. Correct V (Y-component)
    // velocity[1] vive a j+1/2. Usa p[j+1] - p[j]
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N - 1; ++j) { // Nota il limite N-1
            for (int k = 0; k < N; ++k) {
                int idx = Idx(i, j, k, N);
                int idx_next = Idx(i, j + 1, k, N);
                
                double dp_dy = (pressure[idx_next] - pressure[idx]) / dx;
                velocity[1][idx] -= dt * dp_dy;
            }
        }
    }

    // 3. Correct W (Z-component)
    // velocity[2] vive a k+1/2. Usa p[k+1] - p[k]
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N - 1; ++k) { // Nota il limite N-1
                int idx = Idx(i, j, k, N);
                int idx_next = Idx(i, j, k + 1, N);
                
                double dp_dz = (pressure[idx_next] - pressure[idx]) / dx;
                velocity[2][idx] -= dt * dp_dz;
            }
        }
    }
}


void NavierStokesBrinkmann::compute_velocity_divergence(double time) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                int idx = Idx(i, j, k, N);

                // Grid Coordinates of the Pressure Node
                double x = i * dx;
                double y = j * dx;
                double z = k * dx;

                // ---------------------------------------------------------
                // 1. X-Divergence (du/dx)
                // Right Face: Internal node (valid for i < N-1), or Boundary Value (i = N-1)
                // Left Face: Internal node (i > 0), or Ghost Value (i = 0)
                
                double u_plus = velocity[0][Idx(i, j, k, N)];
                double u_minus;

                if (i > 0) {
                    u_minus = velocity[0][Idx(i - 1, j, k, N)];
                } else {
                    // LEFT WALL (x=0):
                    // Pressure is at x=0. The velocity flux "entering" this control volume
                    // must be evaluated at x = -dx/2.
                    u_minus = exact_solution.value(0, x - dx/2.0, y, z, time);
                }
                
                double dudx = (u_plus - u_minus) / dx;

                // ---------------------------------------------------------
                // 2. Y-Divergence (dv/dy)
                
                double v_plus = velocity[1][Idx(i, j, k, N)];
                double v_minus;

                if (j > 0) {
                    v_minus = velocity[1][Idx(i, j - 1, k, N)];
                } else {
                    // BOTTOM WALL (y=0): Ghost value at y = -dx/2
                    v_minus = exact_solution.value(1, x, y - dx/2.0, z, time);
                }

                double dvdy = (v_plus - v_minus) / dx;

                // ---------------------------------------------------------
                // 3. Z-Divergence (dw/dz)

                double w_plus = velocity[2][Idx(i, j, k, N)];
                double w_minus;

                if (k > 0) {
                    w_minus = velocity[2][Idx(i, j, k - 1, N)];
                } else {
                    // BACK WALL (z=0): Ghost value at z = -dx/2
                    w_minus = exact_solution.value(2, x, y, z - dx/2.0, time);
                }

                double dwdz = (w_plus - w_minus) / dx;

                // ---------------------------------------------------------
                // RHS = - Div(u) / dt
                rhs[idx] = -(dudx + dvdy + dwdz) / dt;
            }
        }
    }
}

void NavierStokesBrinkmann::get_error(double &total_error_vel, double &total_error_pres, double current_time){

        // Compute error norms between numerical and exact solutions
    double error_u = 0.0;
    double error_v = 0.0;
    double error_w = 0.0;
    double error_p = 0.0;
    double vel_norm = 0.0;
    double pres_norm = 0.0;
    double vel_norm_comparison = 0.0;
    double pres_norm_comparison = 0.0;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                double x = i * dx;
                double y = j * dx;
                double z = k * dx;

                int idx = Idx(i, j, k, N);

                double u_exact = exact_solution.value(0, x+dx/2, y, z, current_time);
                double v_exact = exact_solution.value(1, x, y+dx/2, z, current_time);
                double w_exact = exact_solution.value(2, x, y, z+dx/2, current_time);
                double p_exact = exact_solution.value(3, x, y, z, current_time);

                error_u += (velocity[0][idx] - u_exact) * (velocity[0][idx] - u_exact);
                error_v += (velocity[1][idx] - v_exact) * (velocity[1][idx] - v_exact);
                error_w += (velocity[2][idx] - w_exact) * (velocity[2][idx] - w_exact);
                error_p += (pressure[idx] - p_exact) * (pressure[idx] - p_exact);

                double u_exact1 = exact_solution.value(0, x+dx/2, y, z, 1.0);
                double v_exact1 = exact_solution.value(1, x, y+dx/2, z, 1.0);
                double w_exact1 = exact_solution.value(2, x, y, z+dx/2, 1.0);
                double p_exact1 = exact_solution.value(3, x, y, z, 1.0);


                

                vel_norm_comparison += u_exact1 * u_exact1 + v_exact1 * v_exact1 + w_exact1 * w_exact1;
                pres_norm_comparison += p_exact1 * p_exact1;
                vel_norm += u_exact * u_exact + v_exact * v_exact + w_exact * w_exact;
                pres_norm += p_exact * p_exact;
            }
        }
    }

    double dV = dx * dx * dx;

    total_error_vel = std::sqrt((error_u + error_v + error_w)*dV);
    total_error_pres = std::sqrt(error_p * dV);

}


