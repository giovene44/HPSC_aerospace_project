/**
 * Automatic checks of the direction-splitting solver (make check):
 *  1. batched tridiagonal solver (general and uniform matrices, local or distributed over the
 *     MPI line communicators) against a plain Thomas solve of the global lines;
 *  2. manufactured-solution convergence (space-time, dt ~ h): second order for the velocity
 *     and for the pressure, and the solution does not depend on the process grid.
 * Run serially or with mpirun -n P (any P that leaves at least 2 nodes per process).
 */
#include "Parallel.hpp"
#include "TridiagonalSolver.hpp"
#include "navier_stokes_brinkman.hpp"

#include <cmath>
#include <cstdio>
#include <vector>
#ifdef USE_MPI
#include <mpi.h>
#endif

namespace
{
int failures = 0;
int rank0 = 1;

void check(bool ok, const char *what, double value)
{
    if (rank0)
        std::printf("  [%s] %-58s %.3e\n", ok ? " OK " : "FAIL", what, value);
    if (!ok)
        ++failures;
}

// global test matrix and rhs (functions of the global indices only)
Real ga(int G, int l) { return G == 0 ? 0 : -0.3 - 0.01 * ((G + 3 * l) % 7); }
Real gc(int G, int N, int l) { return G == N - 1 ? 0 : -0.25 - 0.02 * ((2 * G + l) % 5); }
Real gb(int G, int l) { return 1.7 + 0.05 * ((G * l) % 3); }
Real gd(int G, int l) { return std::sin(0.37 * G + 0.11 * l) + 0.2; }

void thomas(std::vector<Real> a, std::vector<Real> b, std::vector<Real> c, std::vector<Real> &d)
{
    const int n = static_cast<int>(d.size());
    c[0] /= b[0];
    d[0] /= b[0];
    for (int i = 1; i < n; ++i)
    {
        const Real m = Real(1) / (b[i] - a[i] * c[i - 1]);
        c[i] *= m;
        d[i] = (d[i] - a[i] * d[i - 1]) * m;
    }
    for (int i = n - 2; i >= 0; --i)
        d[i] -= c[i] * d[i + 1];
}

void test_tridiagonal(const Topology &topo)
{
    const std::array<Dim, 3> N{23, 17, 11};
    Decomposition dec(topo, N);
    TridiagonalSolver solver;
    for (int d = 0; d < 3; ++d)
    {
        const int m = dec.n(d), nl = 5, np = 3, g0 = dec.begin(d), Nd = dec.N(d);
        for (int uniform = 0; uniform < 2; ++uniform)
        {
            std::vector<Real> x(static_cast<std::size_t>(np) * m * nl);
            auto gather = [&](int p, Real *D, Real *)
            {
                for (int g = 0; g < m; ++g)
                    for (int l = 0; l < nl; ++l)
                        D[g * nl + l] = gd(g0 + g, l + nl * p);
            };
            auto scatter = [&](int p, const Real *D)
            {
                for (int i = 0; i < m * nl; ++i)
                    x[static_cast<std::size_t>(p) * m * nl + i] = D[i];
            };
            auto lane = [&](int p, int l) { return uniform ? 0 : l + nl * p; };
            if (uniform)
            {
                std::vector<Real> a(m), b(m), c(m);
                for (int g = 0; g < m; ++g)
                {
                    a[g] = ga(g0 + g, 0);
                    b[g] = gb(g0 + g, 0);
                    c[g] = gc(g0 + g, Nd, 0);
                }
                UniformMatrix M;
                M.set(dec, d, a, b, c);
                solver.sweep_uniform(dec, d, np, nl, M, gather, scatter);
            }
            else
            {
                auto coef = [&](int p, int g, const Real *, Real *A, Real *B, Real *C)
                {
                    for (int l = 0; l < nl; ++l)
                    {
                        A[l] = ga(g0 + g, lane(p, l));
                        B[l] = gb(g0 + g, lane(p, l));
                        C[l] = gc(g0 + g, Nd, lane(p, l));
                    }
                };
                solver.sweep(dec, d, np, m, nl, gather, coef, scatter);
            }
            double err = 0;
            for (int p = 0; p < np; ++p)
                for (int l = 0; l < nl; ++l)
                {
                    std::vector<Real> a(Nd), b(Nd), c(Nd), r(Nd);
                    for (int G = 0; G < Nd; ++G)
                    {
                        a[G] = ga(G, lane(p, l));
                        b[G] = gb(G, lane(p, l));
                        c[G] = gc(G, Nd, lane(p, l));
                        r[G] = gd(G, l + nl * p);
                    }
                    thomas(a, b, c, r);
                    for (int g = 0; g < m; ++g)
                        err = std::max(err, std::fabs(double(x[static_cast<std::size_t>(p) * m * nl + g * nl + l] - r[g0 + g])));
                }
            err = topo.max(err);
            char what[96];
            std::snprintf(what, sizeof what, "tridiagonal %s, direction %d (P=%d): max error", uniform ? "uniform" : "general", d,
                          dec.P(d));
            check(err < 1e-13, what, err);
        }
    }
}

ProblemData mms_problem(const std::string &k, Real nu, Real chi)
{
    // u = sin t (sx sy sz, cx cy cz, cx sy (sz + cz)), p = sin t sx sy sz (div u = 0)
    // f = du/dt + nu/k u - nu Lap u + grad p, with Lap u = -3u.
    ProblemData pd;
    pd.L = {Real(2 * M_PI), Real(2 * M_PI), Real(2 * M_PI)};
    pd.nu = nu;
    pd.k_expr = k;
    pd.chi = chi;
    const std::string U[3] = {"sin(x)*sin(y)*sin(z)", "cos(x)*cos(y)*cos(z)", "cos(x)*sin(y)*(sin(z)+cos(z))"};
    const std::string G[3] = {"cos(x)*sin(y)*sin(z)", "sin(x)*cos(y)*sin(z)", "sin(x)*sin(y)*cos(z)"};
    const std::string snu = std::to_string(nu);
    for (int c = 0; c < 3; ++c)
    {
        pd.u_exact[c] = "sin(t)*" + U[c];
        pd.u_boundary[c] = pd.u_exact[c];
        pd.forcing[c] = "cos(t)*" + U[c] + " + " + snu + "/(" + k + ")*sin(t)*" + U[c] + " + 3*" + snu + "*sin(t)*" +
                        U[c] + " + sin(t)*" + G[c];
    }
    pd.p_exact = "sin(t)*sin(x)*sin(y)*sin(z)";
    return pd;
}

ErrorNorms run(const Topology &topo, const ProblemData &pd, Dim N, Real dt, Real T)
{
    Decomposition dec(topo, {N, N, N});
    NavierStokesBrinkman s(pd, dec, dt, Real(1.5));
    s.initialize();
    s.advance(static_cast<int>(std::lround(T / dt)));
    return s.errors();
}

void test_convergence(const Topology &topo, const char *label, const ProblemData &pd, double min_rate_p)
{
    ErrorNorms e[3];
    const Dim N0 = 8;
    for (int r = 0; r < 3; ++r)
        e[r] = run(topo, pd, N0 << r, Real(0.1) / (1 << r), Real(1));
    const double hr = std::log((N0 * 4 - 0.5) / (N0 * 2 - 0.5)); // h ratio between N=16 and N=32
    const double ru = std::log(e[1].u_l2 / e[2].u_l2) / hr, rp = std::log(e[1].p_l2 / e[2].p_l2) / hr;
    char what[96];
    std::snprintf(what, sizeof what, "%s: velocity rate (N=16->32, dt~h)", label);
    check(ru > 1.8, what, ru);
    std::snprintf(what, sizeof what, "%s: pressure rate (N=16->32, dt~h)", label);
    check(rp > min_rate_p, what, rp);
    std::snprintf(what, sizeof what, "%s: relative velocity error N=32", label);
    check(e[2].u_l2_rel < 3e-3, what, e[2].u_l2_rel);
}
} // namespace

int main(int argc, char **argv)
{
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
#else
    (void)argc;
    (void)argv;
#endif
    {
        Topology topo;
        rank0 = topo.rank() == 0;
        if (rank0)
            std::printf("Solver checks on %d process(es): %d x %d x %d\n", topo.size(), topo.P(0), topo.P(1), topo.P(2));
        test_tridiagonal(topo);
        test_convergence(topo, "k=1, nu=0.1, standard", mms_problem("1", 0.1, 0), 1.6);
        test_convergence(topo, "k=1, nu=0.1, rotational", mms_problem("1", 0.1, 1), 1.9);
        test_convergence(topo, "k variable, nu=1, rotational", mms_problem("1/(2+sin(x)*sin(y)*sin(z))", 1.0, 1), 1.9);
        // reference values computed with 1 process: the result must not depend on P
        const ErrorNorms e = run(topo, mms_problem("1", 0.1, 1), 16, 0.05, 1.0);
        check(std::fabs(e.u_l2 - 5.02238e-02) < 1e-6 && std::fabs(e.p_l2 - 5.293586e-02) < 1e-6,
              "N=16 errors equal to the single-process run", std::fabs(e.u_l2 - 5.02238e-02));
        if (rank0)
            std::printf("%s\n", failures ? "SOME CHECKS FAILED" : "all checks passed");
    }
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return failures ? 1 : 0;
}
