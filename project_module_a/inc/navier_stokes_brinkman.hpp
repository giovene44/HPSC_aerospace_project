#include <string>

class NavierStokesBrinkmann {
public:
    class Function{
    public:
        Function(){}
        float value(/*Point*/){};
    };

    class Porosity{}; // TODO: Discuss in case the porosity is in a file or is a given function

    class ForcingTerm : public Function{
    public:
        ForcingTerm(){}
        float value(int i, int j, int k, float t){ return 0.0; } // Placeholder implementation
    };

    
    class Nu: public Function{};

    class Grid{

    // Until we get more information for this part we will leave it empty
    public:
        Grid(){}

        void compute_grid(){};
        void split_in_blocks(){};
        void compute_porosity(/*data structure for porosity*/){};

        protected:
    };

    NavierStokesBrinkmann();
    void solve();
protected:

    //setup methods:
    void setup(); //read grid, setup initial values, assemble K, gamma, initial conditions

    //methods inside the iteration:
    void assemble_g_rhs();
    void solve_momentum();
    void solve_pressure();
    void output_results(int timestep) const;


    Grid grid;
    Porosity porosity;
    ForcingTerm forcing_term;

    //Some of these data structures can be merged in one with further optimization

    /*Data structure*/ auto pressure_predictor;
    /*Data structure*/ auto velocity_predictor;
    /*Data structure*/ auto psi;
    /*Data structure*/ auto phi;
    /*Data structure*/ auto other_phi;
    /*Data structure*/ auto gradient_pressure; // TODO: Choose if separate the components or not
    /*Data structure*/ auto g_rhs; // rhs of the momentum equation
    /*Data structure*/ auto eta;
    /*Data structure*/ auto zeta;
    /*Data structure*/ auto ksi;
    /*Data structure*/ auto nu;
    /*Data structure*/ auto beta;
    /*Data structure*/ auto gamma;


    /*Data structure*/ auto velocity_solution;
    /*Data structure*/ auto pressure_solution;
};