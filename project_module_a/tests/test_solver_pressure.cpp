#include "Solver.hpp"
#include <cmath>
#include <iostream>
#include <vector>
#include <string>

int main()
{
    // DOMINIO: [0, 2pi]^3 per soddisfare Neumann con i coseni
    Real Lx = 2.0 * M_PI;
    Real Ly = 2.0 * M_PI;
    Real Lz = 2.0 * M_PI;

    // TEMPI
    Real t_n_minus_half = 0.5; // p_old
    Real dt = 0.1;
    Real t_n_plus_half = t_n_minus_half + dt; // p_target

    std::vector<Dim> Ns = {16, 32};
    std::vector<Real> phi_errors;
    std::vector<Real> p_errors;

    

    for (const auto &N : Ns)
    {
        Real dx = Lx / (N - 0.5); // Staggered correction se necessaria, o standard
        Real dy = Ly / (N - 0.5);
        Real dz = Lz / (N - 0.5);

        // Oggetti Variabili
        ScalarVariable p_old(N, N, N, dx, dy, dz);
        ScalarVariable p_target(N, N, N, dx, dy, dz);
        ScalarVariable phi_exact(N, N, N, dx, dy, dz);
        ScalarVariable rhs_for_solver(N, N, N, dx, dy, dz);

        // Variabili temporanee per la cascata del solver
        ScalarVariable temp_psi(N, N, N, dx, dy, dz);    // Dopo step X
        ScalarVariable temp_varphi(N, N, N, dx, dy, dz); // Dopo step Y
        ScalarVariable phi_numeric(N, N, N, dx, dy, dz); // Risultato finale (step Z)
        ScalarVariable p_final_numeric(N, N, N, dx, dy, dz);

        // Costante arbitraria della funzione
        Real Amplitude = 3.0 * 1e-1;

        // 1. RIEMPIMENTO DATI ESATTI
        for (Dim i = 0; i < N; ++i)
        {
            for (Dim j = 0; j < N; ++j)
            {
                for (Dim k = 0; k < N; ++k)
                {
                    Real x = i * dx;
                    Real y = j * dy;
                    Real z = k * dz;

                    // Funzione spaziale base: cos(x)cos(y)cos(z)
                    // NOTA: Usiamo coseni su tutte le direzioni per soddisfare Neumann (derivata nulla ai bordi 0 e 2pi)
                    Real space_part = std::cos(x) * std::cos(y) * std::cos(z);

                    // p al tempo n-1/2
                    p_old.set(i, j, k) = std::sin(t_n_minus_half) * Amplitude * space_part;

                    // p al tempo n+1/2 (Target)
                    p_target.set(i, j, k) = std::sin(t_n_plus_half) * Amplitude * space_part;

                    // phi esatto = p_new - p_old
                    phi_exact.set(i, j, k) = p_target.get(i, j, k) - p_old.get(i, j, k);

                    // CALCOLO RHS ANALITICO PER IL SOLVER
                    // L'operatore è (I - dxx)(I - dyy)(I - dzz)
                    // Poiché dxx(cos x) = -cos x, allora (I - dxx) diventa un fattore (1 - (-1)) = 2.
                    // Stessa cosa per y e z.
                    // Totale operatore = 2 * 2 * 2 = 8.
                    // Quindi il Solver deve risolvere: Operator * phi = RHS
                    // => 8 * phi_exact = RHS
                    rhs_for_solver.set(i, j, k) = 8.0 * phi_exact.get(i, j, k);
                }
            }
        }

        // 2. SETUP SOLVER
        // Boundary fittizio (non usato nel calcolo interno se passiamo il RHS giusto,
        // ma necessario per costruttore). Impostiamo a 0.0.
        std::vector<std::string> zero_bc = {"0.0", "0.0", "0.0"};
        BoundaryFunctions bc_func;
        bc_func.set_string_expression(zero_bc);

        PressureSolver solver(N, N, N, dx, dy, dz, dt, bc_func);
        DimensionsHandlerScalar dim_handler(N, N, N, dx);

        // 3. ESECUZIONE CASCATA (Direction Splitting)

        // Step 1: Risolve lungo X -> ottiene psi
        // Equation: (I - dxx) psi = RHS
        solver.solve_pressure<0>(rhs_for_solver, temp_psi, dim_handler);

        // Step 2: Risolve lungo Y -> ottiene varphi
        // Equation: (I - dyy) varphi = psi
        solver.solve_pressure<1>(temp_psi, temp_varphi, dim_handler);

        // Step 3: Risolve lungo Z -> ottiene phi (correzione finale)
        // Equation: (I - dzz) phi = varphi
        solver.solve_pressure<2>(temp_varphi, phi_numeric, dim_handler);

        // 4. AGGIORNAMENTO PRESSIONE
        // p_new = p_old + phi
        for (size_t idx = 0; idx < p_old.size(); ++idx)
        {
            p_final_numeric.set(idx) = p_old.get(idx) + phi_numeric.get(idx);
        }

        // 5. STAMPA RISULTATI
        std::cout << "--- Grid " << N << "^3 ---" << std::endl;
        // Verifica se phi è stato calcolato correttamente (test del solver puro)
        Real abs_error = 0.0;
        Real norm = 0.0;
        for (size_t i = 0; i < phi_exact.size(); ++i)
        {
            Real diff = phi_exact.get(i) - phi_numeric.get(i);
            abs_error += diff * diff;
            norm += phi_exact.get(i) * phi_exact.get(i);
        }
        abs_error = std::sqrt(abs_error);
        norm = std::sqrt(norm);

        phi_errors.push_back(abs_error / (norm > 1e-15 ? norm : 1.0));

        std::cout << "Phi Solver Accuracy" << " | Rel Error: " << (norm > 1e-15 ? abs_error / norm : 0.0) << std::endl;
        // Verifica se la pressione finale corrisponde al target (test dell'update)
        abs_error = 0.0;
        norm = 0.0;
        for (size_t i = 0; i < p_target.size(); ++i)
        {
            Real diff = p_target.get(i) - p_final_numeric.get(i);
            abs_error += diff * diff;
            norm += p_target.get(i) * p_target.get(i);
        }
        abs_error = std::sqrt(abs_error);
        norm = std::sqrt(norm);
        p_errors.push_back(abs_error / (norm > 1e-15 ? norm : 1.0));
        std::cout << "Pressure Update Accuracy"
                  << " | Rel Error: " << (norm > 1e-15 ? abs_error / norm : 0.0) << std::endl;
        std::cout << std::endl;
    }

    // Compute convergence rates
    if (phi_errors.size() >= 2) {
        std::cout << "--- Convergence Rates ---" << std::endl;
        for (size_t i = 1; i < phi_errors.size(); ++i) {
            Real rate_phi = std::log(phi_errors[i-1] / phi_errors[i]) / std::log(2.0);
            Real rate_p = std::log(p_errors[i-1] / p_errors[i]) / std::log(2.0);
            std::cout << "Refinement " << i << " (N=" << Ns[i-1] << " -> " << Ns[i] << ")" << std::endl;
            std::cout << "  Phi convergence rate: " << rate_phi << std::endl;
            std::cout << "  Pressure convergence rate: " << rate_p << std::endl;
        }
    }

    return 0;
}