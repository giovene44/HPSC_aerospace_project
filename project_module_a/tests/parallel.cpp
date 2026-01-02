#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
#include "navier_stokes_brinkman.hpp"
#include <chrono>

// ===============================================================
// 1. Setup Grid Parameters
// ===============================================================
struct Grid
{
    Dim Nx, Ny, Nz;
    Real dx, dy, dz;
    Real dt;
};

Grid setup_grid(Real dim_x, Real dim_y, Real dim_z, Dim Nx, Dim Ny, Dim Nz, Real dt)
{
    Grid g;
    g.Nx = Nx;
    g.Ny = Ny;
    g.Nz = Nz;
    g.dx = dim_x / (Nx - 0.5);
    g.dy = dim_y / (Ny - 0.5);
    g.dz = dim_z / (Nz - 0.5);
    g.dt = dt;
    return g;
}

/**
 * @brief Converte indici locali in globali per una griglia arbitraria di processi.
 * * @param i_L Indice di riga locale nel sottodominio.
 * @param j_L Indice di colonna locale nel sottodominio.
 * @param px Indice del processo lungo l'asse X (0, 1, 2...).
 * @param py Indice del processo lungo l'asse Y (0, 1, 2...).
 * @param paddle Dimensione del salto tra l'inizio di un sottodominio e il successivo.
 */
std::vector<int> local_to_global_grid(int i_L, int j_L, int px, int py, int paddle) {
    std::vector<int> indexes_glob(2, 0);

    // L'indice globale è dato dall'offset del processo + l'indice locale
    // Nel tuo codice main, l'offset è calcolato come px * paddle_direction
    int I_G = i_L + (px * paddle);
    int J_G = j_L + (py * paddle);

    indexes_glob[0] = I_G;
    indexes_glob[1] = J_G;

    return indexes_glob;
}


//N = numero di punti del blocco 
//I_G = indice globale 

std::vector<double> global_to_local(int N_R, int N_C, int I_G, int J_G) { 
    // La matrice globale M ha dimensione 2N x 2N
    
    std::vector<double> indexes_loc(2,0); 
    std::cout << " Indice Globale : " << I_G << " " << J_G << std::endl; 

    int i_L = 0; 
    int j_L = 0;

    // Calcolo i_L (Indice di Riga Locale) - Usa N_R
    if (I_G < N_R) {
        // Blocco A o B (righe da 0 a N_R-1)
        i_L = I_G;
    } else {
        // Blocco C o D (righe da N_R a 2N_R-1)
        i_L = I_G - N_R; // Applica l'offset di riga
    }

    // Calcolo j_L (Indice di Colonna Locale) - Usa N_C
    if (J_G < N_C) {
        // Blocco A o C (colonne da 0 a N_C-1)
        j_L = J_G;
    } else {
        // Blocco B o D (colonne da N_C a 2N_C-1)
        j_L = J_G - N_C; // Applica l'offset di colonna
    }

    std::cout << " Indice Locale : " << i_L << " " << j_L << std::endl; 
    indexes_loc[0] = i_L; 
    indexes_loc[1] = j_L; 
    return  indexes_loc; 
}


void test_global_to_Local_function(){
    int N = 11;             // Numero di punti lungo asse x 
    int N_points = N * N;   // Numero totale di punti della componete u_x della velocità in 2D 
    int Num_processes = 3;  // Numero di processi che uso per la componente u_x 
    int Num_points_subdomain = int(N / Num_processes) + 1;  //Numero di intrni punti di ogni sotto dominio lungo un asse 

    int Num_pints_shared =  N % 3; 
    int val = 1; 
    int idx = 0; 
    int idy = 0; 
    int k = 1; 

    std::vector<double> indexes_local; 
    std::vector<std::vector<double>> M(N,std::vector<double>(N,0));

    std::vector<std::vector<double>> A_loc(Num_points_subdomain, std::vector<double>(Num_points_subdomain,1)); 
    std::vector<std::vector<double>> B_loc(Num_points_subdomain, std::vector<double>(Num_points_subdomain+1,2)); 

    for(int i = 0; i < Num_points_subdomain; i++){
        A_loc[i][3] = 0; 
        A_loc[3][i] = 0; 
    }

    for(int i = 0; i < Num_points_subdomain; i++){
        B_loc[i][0] = 0; 
        B_loc[3][i] = 0; 
        B_loc[i][4] = 0; 
    }

    std::cout << "Num processi : "<< Num_processes << std::endl; 
    std::cout << "Num points totali matrice : "<< N_points << std::endl; 
    std::cout << "Num points per ogni sottodominio  : "<< Num_points_subdomain << std::endl;  

    for(int j = 0; j < N; j++ ){ 
        val = k; 
        idx = 0;  
        for(int i = 0; i < N; i++){
            if ( idx  < Num_points_subdomain -1  && idy != Num_points_subdomain - 1){
                M[j][i] = val; 
                idx++;   
            }
            else{ 
                idx = 0; 
                M[j][i] = 0; 
                val++; 
            }
            std::cout << M[j][i] << ' ';
        }
        if( idy == Num_points_subdomain - 1){
            idy = 0;  
            k = k + Num_points_subdomain- 1 ; 
        }
        else{
            idy++; 
        }
        std::cout << std::endl; 
    }

    indexes_local = global_to_local(Num_points_subdomain,Num_points_subdomain,3,2); 

    std::cout << "Matrice A_loc " << std::endl; 

    for(int j = 0; j < Num_points_subdomain; j++){
        for(int i = 0; i < Num_points_subdomain; i++){
            std::cout << A_loc[j][i] << " "; 
        }
        std::cout << std::endl; 
    }

    std::cout <<  " Valore di A_loc " << A_loc[indexes_local[0]][indexes_local[1]] << std::endl; 


    indexes_local = global_to_local(Num_points_subdomain,Num_points_subdomain+1,6,7); 

    std::cout << "Matrice B_loc " << std::endl; 

    for(int j = 0; j < Num_points_subdomain; j++){
        for(int i = 0; i < Num_points_subdomain + 1 ; i++){
            std::cout << B_loc[j][i] << " "; 
        }
        std::cout << std::endl; 
    }

    std::cout <<  " Valore di B_loc " << B_loc[indexes_local[0]][indexes_local[1]] << std::endl; 
    std::cout << "Funz" << std::endl; 
}

void test_conversion_logic() {
    int paddle = 4; // Come definito nel tuo main
    
    // Supponiamo di essere nel Processo (px=1, py=2)
    // E di avere un indice locale (i_L=1, j_L=1)
    int px = 1; 
    int py = 2;
    int i_L = 1;
    int j_L = 1;

    std::vector<int> glob = local_to_global_grid(i_L, j_L, px, py, paddle);

    std::cout << "--- Test Griglia Processi ---" << std::endl;
    std::cout << "Processo: [" << px << "][" << py << "]" << std::endl;
    std::cout << "Locale: (" << i_L << "," << j_L << ")" << std::endl;
    std::cout << "Globale Risultante: (" << glob[0] << "," << glob[1] << ")" << std::endl;
    
    // Verifica: I_G = 1 + (1 * 4) = 5; J_G = 1 + (2 * 4) = 9
    if(glob[0] == 5 && glob[1] == 9) {
        std::cout << "SUCCESS: Conversione corretta." << std::endl;
    } else {
        std::cout << "FAILURE: Errore nel calcolo." << std::endl;
    }
}


// ===============================================================
// 3. Initialize Fields (interior + RHS)
// ===============================================================
void initialize_process_fields(const Grid &g,
                               ScalarVariable &gamma_field,
                               VectorVariable &vector,
                               VectorVariable &rhs,
                               BoundaryFunctions &u_boundary,
                               Real t,
                               int start_index_i_global,
                               int start_index_j_global,
                               int start_index_k_global,
                               int end_index_i_global,
                               int end_index_j_global,
                               int end_index_k_global,
                               int start_index_i_local,
                               int start_index_j_local,
                               int start_index_k_local)
{
    (void)u_boundary; // Unused parameter

    printf("  initialize_process_fields: global range [%d,%d)x[%d,%d)x[%d,%d), local start (%d,%d,%d)\n",
           start_index_i_global, end_index_i_global,
           start_index_j_global, end_index_j_global,
           start_index_k_global, end_index_k_global,
           start_index_i_local, start_index_j_local, start_index_k_local);
    printf("  vector dimensions: %d x %d x %d\n", vector.get_Nx(), vector.get_Ny(), vector.get_Nz());

    // Initialize interior
    for (int comp = 0; comp < 3; ++comp)
        for (int k = start_index_k_global; k < end_index_k_global; ++k)
            for (int j = start_index_j_global; j < end_index_j_global; ++j)
                for (int i = start_index_i_global; i < end_index_i_global; ++i)
                {
                    Real x, y, z;
                    if (comp == 0)
                    {
                        x = (i + 0.5) * g.dx;
                        y = j * g.dy;
                        z = k * g.dz;
                    }
                    else if (comp == 1)
                    {
                        x = i * g.dx;
                        y = (j + 0.5) * g.dy;
                        z = k * g.dz;
                    }
                    else
                    {
                        x = i * g.dx;
                        y = j * g.dy;
                        z = (k + 0.5) * g.dz;
                    }

                    // Convert global index to local index
                    int i_local = i - start_index_i_global + start_index_i_local;
                    int j_local = j - start_index_j_global + start_index_j_local;
                    int k_local = k - start_index_k_global + start_index_k_local;

                    // Debug: check bounds
                    if (i_local < 0 || i_local >= vector.get_Nx() ||
                        j_local < 0 || j_local >= vector.get_Ny() ||
                        k_local < 0 || k_local >= vector.get_Nz())
                    {
                        printf("  ERROR: Out of bounds access! global(%d,%d,%d) -> local(%d,%d,%d), vector size (%d,%d,%d)\n",
                               i, j, k, i_local, j_local, k_local, vector.get_Nx(), vector.get_Ny(), vector.get_Nz());
                        exit(1);
                    }

                    if (comp == 0)
                        vector.set(comp, i_local, j_local, k_local) = 1;
                    else if (comp == 1)
                        vector.set(comp, i_local, j_local, k_local) = 1;
                    else
                        vector.set(comp, i_local, j_local, k_local) = 1;
                }
    printf("  initialize_process_fields: completed successfully\n");
}

// ===============================================================
// 4. Setup Solver & Strides
// ===============================================================
VelocitySolver setup_solver(const Grid &g, ScalarVariable &gamma_field, BoundaryFunctions &u_boundary, Real dt, Real t)
{
    VelocitySolver solver(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz, dt, gamma_field, u_boundary);
    solver.gamma_field = gamma_field;
    solver.u_boundary = u_boundary;
    solver.set_t(t);
    return solver;
}

auto define_stride_x(const Grid &g)
{
    return [=](Dim j, Dim k)
    { return j * g.Nx + k * g.Nx * g.Ny; };
}

// ===============================================================
// 6. Solve and check solution
// ===============================================================
Real compute_L2_error(VectorVariable &expected, VectorVariable &computed, const Grid &g)
{
    Real error = 0.0;
    for (int comp = 0; comp < 3; ++comp)
        for (Dim k = 0; k < g.Nz; ++k)
            for (Dim j = 0; j < g.Ny; ++j)
                for (Dim i = 0; i < g.Nx; ++i)
                {
                    Real diff = expected.value(comp, i, j, k) - computed.value(comp, i, j, k);
                    error += diff * diff;
                }
    return sqrt(error * g.dx * g.dy * g.dz);
}

bool solve_and_check(VelocitySolver &solver, VectorVariable &rhs_1,
                     VectorVariable &vector_1, VectorVariable &rhs_2,
                     VectorVariable &vector_2, const Grid &g,
                     DimensionsHandlerVector &x_handler,
                     Real &l2_error, auto &time_speedup)
{
    // PARALLEL SOLVE
    VectorVariable rhs_delta_parallel(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    VectorVariable vector_delta_parallel(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    rhs_delta_parallel = rhs_2 - rhs_1;
    vector_delta_parallel = vector_2 - vector_1;
    VectorVariable computed_sol_parallel(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    computed_sol_parallel.set_all(0.0);
    auto time_parallel_start = std::chrono::high_resolution_clock::now();
    solver.solve<0>(rhs_delta_parallel, computed_sol_parallel, x_handler, true);
    auto time_parallel_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_parallel_diff = time_parallel_end - time_parallel_start;
    // Compute L2 error
    l2_error = compute_L2_error(vector_delta_parallel, computed_sol_parallel, g);

    // SERIAL SOLVE
    VectorVariable rhs_delta_serial(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    VectorVariable vector_delta_serial(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    rhs_delta_serial = rhs_2 - rhs_1;
    vector_delta_serial = vector_2 - vector_1;
    VectorVariable computed_sol_serial(g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz);
    computed_sol_serial.set_all(0.0);
    auto time_serial_start = std::chrono::high_resolution_clock::now();
    solver.solve<0>(rhs_delta_serial, computed_sol_serial, x_handler, false);
    auto time_serial_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_serial_diff = time_serial_end - time_serial_start;

    time_speedup = time_serial_diff.count() / time_parallel_diff.count();

    return true;
}

// Helper function to plot VectorVariable fields
auto plot_vector_fields = [](const std::vector<std::vector<std::vector<VectorVariable>>> &fields, const Grid &g, const std::string &filename)
{
    std::ofstream file(filename);
    file << "# x y z u v w\n";
    for (int px = 0; px < fields.size(); ++px)
    {
        for (int py = 0; py < fields[px].size(); ++py)
        {
            for (int pz = 0; pz < fields[px][py].size(); ++pz)
            {
                const auto &field = fields[px][py][pz];
                for (int i = 0; i < g.Nx; ++i)
                {
                    for (int j = 0; j < g.Ny; ++j)
                    {
                        for (int k = 0; k < g.Nz; ++k)
                        {
                            Real x = (i + 0.5) * g.dx;
                            Real y = (j + 0.5) * g.dy;
                            Real z = (k + 0.5) * g.dz;
                            file << x << " " << y << " " << z << " "
                                 << field.value(0, i, j, k) << " "
                                 << field.value(1, i, j, k) << " "
                                 << field.value(2, i, j, k) << "\n";
                        }
                    }
                }
            }
        }
    }
    file.close();
};

// ===============================================================
// 7. Main function
// ===============================================================
/*
int main()
{
    const Real two_pi = 2.0 * 3.141592653589793;

    // Open output file
    std::ofstream outfile("convergence_momentum_x.txt");
    outfile << "# Convergence study for momentum x-direction solver\n";
    outfile << "# Nx Ny Nz dx dt L2_error convergence_rate\n";
    outfile << std::scientific << std::setprecision(8);

    // Test multiple grid resolutions
    std::vector<Dim> grid_sizes = {11};
    std::vector<Real> errors;
    std::vector<Real> speed_ups;
    std::vector<Real> dx_values;
    std::vector<Real> dt_values;

    for (Dim N : grid_sizes)
    {
        // Make dt proportional to dx^2 to keep temporal error negligible for spatial convergence study
        Real dx_nominal = two_pi / (N - 0.5);
        // Real dt_test = 0.001 * dx_nominal; // dt ~ O(dx^2)
        Real dt_test = 0.0025;
        Grid g = setup_grid(two_pi, two_pi, two_pi, N, N, N, dt_test);

        printf("\n=================================================\n");
        printf("Grid: Nx=%d, Ny=%d, Nz=%d, dx=%f, dy=%f, dz=%f, dt=%f\n",
               g.Nx, g.Ny, g.Nz, g.dx, g.dy, g.dz, g.dt);
        printf("=================================================\n");

        // dimension of each process is ( 2, 2, 2)    => so we create 4/2=2  processes in each direction
        // !!! N MUST FOLLOW THIS RULE : (N - 2) % num_procs_per_dim == 0
        const int num_procs_per_dim = 3; // Parameter for number of sub-processes per dimension
        int num_point_direction = num_procs_per_dim + 2;
        int paddle_direction = num_point_direction - 1;
        std::vector<std::vector<std::vector<ScalarVariable>>> gamma_fields(
            num_point_direction, std::vector<std::vector<ScalarVariable>>(
                                     num_point_direction, std::vector<ScalarVariable>(
                                                              num_point_direction, ScalarVariable(num_point_direction, num_point_direction, num_point_direction, g.dx, g.dy, g.dz))));
        std::vector<std::vector<std::vector<VectorVariable>>> vector_1(
            num_point_direction, std::vector<std::vector<VectorVariable>>(
                                     num_point_direction, std::vector<VectorVariable>(
                                                              num_point_direction, VectorVariable(num_point_direction, num_point_direction, num_point_direction, g.dx, g.dy, g.dz))));

        std::vector<std::vector<std::vector<VectorVariable>>> rhs_1(
            num_point_direction, std::vector<std::vector<VectorVariable>>(
                                     num_point_direction, std::vector<VectorVariable>(
                                                              num_point_direction, VectorVariable(num_point_direction, num_point_direction, num_point_direction, g.dx, g.dy, g.dz))));
        std::vector<std::vector<std::vector<VectorVariable>>> vector_2(
            num_point_direction, std::vector<std::vector<VectorVariable>>(
                                     num_point_direction, std::vector<VectorVariable>(
                                                              num_point_direction, VectorVariable(num_point_direction, num_point_direction, num_point_direction, g.dx, g.dy, g.dz))));
        std::vector<std::vector<std::vector<VectorVariable>>> rhs_2(
            num_point_direction, std::vector<std::vector<VectorVariable>>(
                                     num_point_direction, std::vector<VectorVariable>(
                                                              num_point_direction, VectorVariable(num_point_direction, num_point_direction, num_point_direction, g.dx, g.dy, g.dz))));

        // set all Gamma to 0.1f
        for (int px = 0; px < num_procs_per_dim; px++)
        {
            for (int py = 0; py < num_procs_per_dim; py++)
            {
                for (int pz = 0; pz < num_procs_per_dim; pz++)
                {
                    gamma_fields[px][py][pz].set_all(0.1f);
                }
            }
        }

        BoundaryFunctions u_boundary;
        std::vector<std::string> sin_bc = {"sin(t)*sin(x)*sin(y)*sin(z)", "sin(t)*cos(x)*cos(y)*cos(z)", "sin(t)*cos(x)*sin(y)*(sin(z)+cos(z))"};
        u_boundary.set_string_expression(sin_bc);

        // Initialize fields for each process subdomain
        printf("Initializing %dx%dx%d subdomains...\n", num_procs_per_dim, num_procs_per_dim, num_procs_per_dim);
        for (int px = 0; px < num_procs_per_dim; px++)
        {
            for (int py = 0; py < num_procs_per_dim; py++)
            {
                for (int pz = 0; pz < num_procs_per_dim; pz++)
                {
                    printf("Process [%d][%d][%d]:\n", px, py, pz);
                    initialize_process_fields(g, gamma_fields[px][py][pz], vector_1[px][py][pz], rhs_1[px][py][pz], u_boundary, g.dt, px * paddle_direction, py * paddle_direction, pz * paddle_direction, (px + 1) * paddle_direction - 1, (py + 1) * paddle_direction - 1, (pz + 1) * paddle_direction - 1, 1, 1, 1);
                    initialize_process_fields(g, gamma_fields[px][py][pz], vector_2[px][py][pz], rhs_2[px][py][pz], u_boundary, g.dt + g.dt, px * paddle_direction, py * paddle_direction, pz * paddle_direction, (px + 1) * paddle_direction - 1, (py + 1) * paddle_direction - 1, (pz + 1) * paddle_direction - 1, 1, 1, 1);
                }
            }
        }

        // plot initial fields
        printf("\nPlotting vector_1 for process [%d][%d][%d]...\n", 0, 0, 0);
        vector_1[0][0][0].print_vector();
    }
    return 0;
}
*/


int main(){
    test_global_to_Local_function(); 
    test_conversion_logic(); 
    return 0; 
}