#include "run_err.hpp"

#include "BoundaryFunctions.hpp"
#include "Parallel.hpp"
#include "ParseInput.hpp"
#include "navier_stokes_brinkman.hpp"

#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef USE_MPI
#include <mpi.h>
#endif

namespace
{
int max_threads()
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}
void set_threads(int n)
{
#ifdef _OPENMP
    omp_set_num_threads(n);
#else
    (void)n;
#endif
}

struct RunResult
{
    Dim N = 0;
    Real h = 0, dt = 0;
    long nsteps = 0;
    ErrorNorms err;
    double time_omp = 0, time_no_omp = 0;
};

struct Options
{
    std::string input = "./Input/Input.in";
    std::array<int, 3> procs{0, 0, 0};
    bool omp_reference = true;
    bool vtk = false;
};

Options parse_options(int argc, char **argv)
{
    Options o;
    for (int a = 1; a < argc; ++a)
    {
        const std::string s = argv[a];
        if ((s == "-i" || s == "--input") && a + 1 < argc)
            o.input = argv[++a];
        else if (s == "--procs" && a + 3 < argc)
        {
            for (int d = 0; d < 3; ++d)
                o.procs[d] = std::atoi(argv[++a]);
        }
        else if (s == "--no-omp-ref")
            o.omp_reference = false;
        else if (s == "--vtk")
            o.vtk = true;
        else
            throw std::runtime_error("Unknown command-line option: " + s);
    }
    return o;
}

ProblemData problem_from_input(const ParseInput &in)
{
    ProblemData pd;
    pd.L = {in.DimX, in.DimY, in.DimZ};
    pd.nu = in.nu;
    pd.k_expr = in.k_expression;
    pd.forcing = {in.fx_expression, in.fy_expression, in.fz_expression};
    pd.u_exact = {in.u_exact_expression, in.v_exact_expression, in.w_exact_expression};
    pd.p_exact = in.p_exact_expression;
    pd.u_boundary = pd.u_exact;
    pd.chi = in.chi;
    // Dirichlet data for the velocity: from the boundary file when it provides 3 expressions
    try
    {
        BoundaryFunctions bf;
        bf.setParsing(in.u_boundary_file);
        if (bf.size() >= 3)
            pd.u_boundary = {bf.string_of(0), bf.string_of(1), bf.string_of(2)};
    }
    catch (const std::exception &)
    {
        // no (valid) boundary file: the exact solution is used as boundary data
    }
    return pd;
}

void make_output_dir()
{
    ::mkdir("OUTPUT", 0755);
}

#ifndef USE_MPI
std::string timestamp()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&t), "%Y-%m-%d_%H-%M-%S");
    return ss.str();
}
#endif
} // namespace

int run_convergence(int argc, char **argv)
{
#ifdef USE_MPI
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
#endif
    int status = 0;
    try
    {
        const Options opt = parse_options(argc, argv);
        Topology topo(opt.procs);
        const bool root = topo.rank() == 0;

        ParseInput &in = ParseInput::getInstance();
        in.parse_input(opt.input);
        const ProblemData pd = problem_from_input(in);
        const int nthreads = max_threads();

        if (root)
        {
            std::cout << "Stokes-Brinkman direction-splitting solver\n"
                      << "  MPI processes : " << topo.size() << " (" << topo.P(0) << " x " << topo.P(1) << " x "
                      << topo.P(2) << ")\n"
                      << "  OpenMP threads: " << nthreads << "\n"
                      << "  nu = " << pd.nu << ", k = " << pd.k_expr << ", t0 = " << in.t0 << ", T = " << in.T
                      << ", dt0 = " << in.dt << ", dt refinement = " << in.dt_refinement
                      << ", grid refinement = " << in.grid_refinement << ", chi = " << in.chi << "\n";
        }

        std::vector<RunResult> results;
        for (int run = 0; run < in.num_runs; ++run)
        {
            RunResult rr;
            Dim f = 1;
            for (int q = 0; q < run; ++q)
                f *= in.grid_refinement;
            const std::array<Dim, 3> N{in.Nx * f, in.Ny * f, in.Nz * f};
            rr.N = N[0];
            rr.dt = in.dt / std::pow(in.dt_refinement, run);
            rr.nsteps = std::max<long>(1, std::lround(in.T / rr.dt));
            Decomposition dec(topo, N);

            auto simulate = [&](int threads, bool keep) -> double
            {
                set_threads(threads);
                NavierStokesBrinkman solver(pd, dec, rr.dt, in.t0);
                solver.initialize();
                const double time = solver.advance(static_cast<int>(rr.nsteps));
                if (keep)
                {
                    rr.err = solver.errors();
                    rr.h = solver.h(0);
                    if (opt.vtk)
                    {
                        make_output_dir();
                        solver.write_vtk("OUTPUT/solution_N" + std::to_string(N[0]));
                    }
                }
                return time;
            };

            if (root)
                std::cout << "\n>>> run " << run + 1 << "/" << in.num_runs << ": N = " << N[0] << " x " << N[1]
                          << " x " << N[2] << ", dt = " << rr.dt << ", steps = " << rr.nsteps << std::endl;

            rr.time_omp = simulate(nthreads, true);
            if (opt.omp_reference && nthreads > 1)
                rr.time_no_omp = simulate(1, false);
            else
                rr.time_no_omp = rr.time_omp;
            set_threads(nthreads);

            if (root)
            {
                std::cout << std::scientific << std::setprecision(4) << "    |u - u_ex|_L2 = " << rr.err.u_l2
                          << " (rel " << rr.err.u_l2_rel << ", max " << rr.err.u_linf << ")\n"
                          << "    |p - p_ex|_L2 = " << rr.err.p_l2 << " (rel " << rr.err.p_l2_rel << ", max "
                          << rr.err.p_linf << ")\n"
                          << "    |div u|_L2    = " << rr.err.div_l2 << "\n"
                          << std::fixed << std::setprecision(3) << "    time = " << rr.time_omp << " s ("
                          << 1e9 * rr.time_omp / (double(rr.nsteps) * N[0] * N[1] * N[2]) << " ns/(node step))";
                if (rr.time_no_omp != rr.time_omp)
                    std::cout << ", 1 thread: " << rr.time_no_omp << " s";
                std::cout << std::defaultfloat << std::endl;
            }
            results.push_back(rr);
        }

        if (root)
        {
            make_output_dir();
#ifdef USE_MPI
            const std::string filename = "OUTPUT/Convergence_Analysis_MPI_" + std::to_string(topo.size()) +
                                         "_OpenMP_" + std::to_string(nthreads) + ".dat";
            const std::string header = "Nx\t\tdx\t\t\tdt\t\t\tnsteps\t\tL2_u_abs\t\tL2_p_abs\t\tL2_u_rel\t\tL2_p_rel\t\t"
                                       "TimeNoOMP\tTimeOMP\t\t\tSpeedUp\t\tProcs\t\tRate_u\t\tRate_p\n";
#else
            const std::string filename = "OUTPUT/Convergence_Analysis_" + timestamp() + ".dat";
            const std::string header =
                "Nx\t\tdx\tdt\t\tnsteps\t\tL2_u_abs\t\tL2_p_abs\t\tL2_u_rel\t\tL2_p_rel\t\tTime\t\tRate_u\t\tRate_p\n";
#endif
            std::ofstream out(filename);
            std::cout << "\n=============================\n   Convergence Analysis\n=============================\n"
                      << header;
            out << header;
            for (std::size_t i = 0; i < results.size(); ++i)
            {
                const RunResult &r = results[i];
                double rate_u = 0, rate_p = 0;
                if (i > 0)
                {
                    // rate with respect to h (or to dt when the grid is not refined)
                    const double lr = (results[i - 1].h != r.h) ? std::log(results[i - 1].h / r.h)
                                                                : std::log(results[i - 1].dt / r.dt);
                    rate_u = std::log(results[i - 1].err.u_l2 / r.err.u_l2) / lr;
                    rate_p = std::log(results[i - 1].err.p_l2 / r.err.p_l2) / lr;
                }
                std::ostringstream oss;
#ifdef USE_MPI
                oss << r.N << "\t\t" << std::scientific << std::setprecision(6) << r.h << "\t\t" << r.dt << "\t\t"
                    << r.nsteps << "\t\t" << r.err.u_l2 << "\t\t" << r.err.p_l2 << "\t\t" << r.err.u_l2_rel << "\t\t"
                    << r.err.p_l2_rel << "\t\t" << r.time_no_omp << "\t" << r.time_omp << "\t\t" << std::fixed
                    << std::setprecision(2) << r.time_no_omp / r.time_omp << "\t\t" << topo.size() << "\t\t" << rate_u
                    << "\t\t" << rate_p << "\n";
#else
                oss << r.N << "\t" << std::scientific << std::setprecision(6) << r.h << "\t" << r.dt << "\t"
                    << r.nsteps << "\t\t" << r.err.u_l2 << "\t\t" << r.err.p_l2 << "\t\t" << r.err.u_l2_rel << "\t\t"
                    << r.err.p_l2_rel << "\t\t" << r.time_omp << "\t\t" << std::fixed << std::setprecision(2) << rate_u
                    << "\t\t" << rate_p << "\n";
#endif
                std::cout << oss.str();
                out << oss.str();
            }
            std::cout << "=============================\nResults saved to " << filename << std::endl;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        status = 1;
#ifdef USE_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
    }
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return status;
}

#ifdef USE_MPI
int run_multiple_mpi(int argc, char **argv) { return run_convergence(argc, argv); }
#else
int run_multiple(int argc, char **argv) { return run_convergence(argc, argv); }
#endif
