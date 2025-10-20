#include <string>
#include "ScalarVariables.hpp"
#include "VectorVariable.hpp"
#include "gamma_beta.h"

class NavierStokesBrinkmann
{
public:
    class Function
    {
    public:
        Function() {}
        float value(/*Point*/){};
    };

    class Porosity
    {
    }; // TODO: Discuss in case the porosity is in a file or is a given function

    class ForcingTerm : public Function
    {
    public:
        ForcingTerm() {}
        float value(int i, int j, int k, float t) { return 0.0; } // Placeholder implementation
    };

    class Nu : public Function
    {
    };

    class Grid
    {

        // Until we get more information for this part we will leave it empty
    public:
        Grid() {}

        void compute_grid() {};
        void split_in_blocks() {};
        void compute_porosity(/*data structure for porosity*/){};

    protected:
    };

    NavierStokesBrinkmann();
    void solve();

protected:
    // setup methods:
    void setup(); // read grid, setup initial values, assemble K, gamma, initial conditions

    // methods inside the iteration:
    void assemble_g();
    auto assemble_rhs(auto &vector_1, auto &vector_2);
    void solve_linear_systems(auto &rhs, auto &a, auto &b, auto &c, auto &output);
    void update_variables();
    void solve_momentum();
    void solve_pressure();
    void output_results(int timestep) const;

    Grid grid;
    Porosity porosity;
    ForcingTerm forcing_term;

    // Some of these data structures can be merged in one with further optimization

    ScalarVariables pressure_predictor;
    VectorVariable velocity_predictor;
    ScalarVariables psi;
    ScalarVariables phi;
    ScalarVariables other_phi;
    ScalarVariables rhs;
    VectorVariable g_rhs; // rhs of the momentum equation
    VectorVariable eta;
    VectorVariable zeta;
    VectorVariable xi;
    VectorVariable nu;
    ScalarVariables sol_linear_system;
    ScalarVariables a;
    ScalarVariables b;
    ScalarVariables c;
    Beta beta;
    Gamma gamma;
    Dim Nx;
    Dim Ny;
    Dim Nz;

    VectorVariable velocity_solution;
    ScalarVariables pressure_solution;
};