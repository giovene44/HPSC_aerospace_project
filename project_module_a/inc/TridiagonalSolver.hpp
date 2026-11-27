#ifndef TRIDIAGONAL_SOLVER_HPP
#define TRIDIAGONAL_SOLVER_HPP

/**
 * @file TridiagonalSolver.hpp
 * @brief Batched solver for the many independent tridiagonal systems produced by the
 *        direction-splitting scheme (one system per grid line).
 *
 * Data layout: the lines of the local block are grouped in "planes"; inside a plane the
 * systems are stored row-major as D[row * nl + lane], so that every elimination step is a
 * loop over the lanes (contiguous, vectorisable) and the recurrence runs over the rows.
 *
 * - Lines owned by a single process (P = 1 along the line): Thomas algorithm.
 * - Lines split over P processes (at least 2 unknowns per process): Schur-complement
 *   (partition) method. Every process
 *   eliminates its interior unknowns with a modified Thomas sweep, so that each local row
 *   only couples to the first and the last local unknowns; these 2P interface unknowns form
 *   a tridiagonal reduced (Schur complement) system that is gathered on every process with
 *   ONE MPI_Allgather for all the lines of the sweep, solved redundantly, and finally the
 *   interior unknowns are recovered locally.
 *
 * Two flavours:
 * - sweep():         coefficients may change from lane to lane; they are generated on the fly
 *                    by coef(plane, row, aux_row, A, B, C) (A x_{row-1} + B x_row + C x_{row+1}).
 * - sweep_uniform(): the matrix is the same for every line (UniformMatrix): its factorisation
 *                    is computed once and the sweep is a division-free substitution.
 * gather(plane, D, AUX) writes the right-hand side (and optionally per-node data in AUX that
 * coef() can read), scatter(plane, D) reads the solution.
 */

#include "Variables.hpp"
#include "Parallel.hpp"
#include <utility>
#include <vector>
#include <cstddef>
#ifdef _OPENMP
#include <omp.h>
#endif

/// Tridiagonal matrix shared by all the lines of a sweep, with its precomputed factors.
class UniformMatrix
{
public:
    /// a, b, c: local rows (sub-, main-, super-diagonal) of the matrix on this process.
    void set(const Decomposition &dec, int d, std::vector<Real> a, std::vector<Real> b, std::vector<Real> c)
    {
        m = static_cast<int>(b.size());
        distributed = dec.P(d) > 1;
        a_ = std::move(a);
        b_ = std::move(b);
        c_ = std::move(c);
        if (!distributed)
        {
            inv.assign(m, 0);
            cp.assign(m, 0);
            inv[0] = Real(1) / b_[0];
            cp[0] = c_[0] * inv[0];
            for (int g = 1; g < m; ++g)
            {
                inv[g] = Real(1) / (b_[g] - a_[g] * cp[g - 1]);
                cp[g] = c_[g] * inv[g];
            }
            return;
        }
        // modified Thomas (see TridiagonalSolver::sweep_distributed), with scalars only
        inv.assign(m, 0);
        A.assign(m, 0);
        C.assign(m, 0);
        bC.assign(m, 0);
        for (int g = 0; g < 2; ++g)
        {
            inv[g] = Real(1) / b_[g];
            A[g] = a_[g] * inv[g];
            C[g] = c_[g] * inv[g];
        }
        for (int g = 2; g < m; ++g)
        {
            inv[g] = Real(1) / (b_[g] - a_[g] * C[g - 1]);
            A[g] = -a_[g] * A[g - 1] * inv[g];
            C[g] = c_[g] * inv[g];
        }
        for (int g = m - 3; g >= 1; --g)
        {
            bC[g] = C[g];
            A[g] -= C[g] * A[g + 1];
            C[g] = -C[g] * C[g + 1];
        }
        if (m > 2) // with m = 2 rows 0 and 1 already couple only to x_0 and x_{m-1}
        {
            r0C = C[0];
            r0inv = Real(1) / (Real(1) - C[0] * A[1]);
            A[0] = A[0] * r0inv;
            C[0] = -C[0] * C[1] * r0inv;
        }
    }

    int m = 0;
    bool distributed = false;
    std::vector<Real> a_, b_, c_;
    // local factors
    std::vector<Real> inv, cp;
    // distributed factors: final couplings to x_0 (A) and x_{m-1} (C), backward multipliers
    std::vector<Real> A, C, bC;
    Real r0C = 0, r0inv = 1;
};

class TridiagonalSolver
{
public:
    template <class Gather, class Coef, class Scatter>
    void sweep(const Decomposition &dec, int d, int np, int m, int nl,
               Gather &&gather, Coef &&coef, Scatter &&scatter)
    {
        if (dec.P(d) == 1)
            sweep_local(np, m, nl, gather, coef, scatter);
        else
            sweep_distributed(dec, d, np, m, nl, gather, coef, scatter);
    }

    template <class Gather, class Scatter>
    void sweep_uniform(const Decomposition &dec, int d, int np, int nl, const UniformMatrix &M,
                       Gather &&gather, Scatter &&scatter)
    {
        if (!M.distributed)
            uniform_local(np, nl, M, gather, scatter);
        else
            uniform_distributed(dec, d, np, nl, M, gather, scatter);
    }

private:
    std::vector<Real> W_, WA_, WC_, send_, recv_, xfl_;

    // ------------------------------------------------------------------ general, local
    template <class Gather, class Coef, class Scatter>
    void sweep_local(int np, int m, int nl, Gather &gather, Coef &coef, Scatter &scatter)
    {
        const std::size_t ps = static_cast<std::size_t>(m) * nl;
#pragma omp parallel
        {
            std::vector<Real> D(ps), CP(ps), AUX(ps), A(nl), B(nl), C(nl);
#pragma omp for schedule(static)
            for (int p = 0; p < np; ++p)
            {
                gather(p, D.data(), AUX.data());
                Real *__restrict d = D.data();
                Real *__restrict cp = CP.data();
                Real *__restrict a = A.data();
                Real *__restrict b = B.data();
                Real *__restrict c = C.data();
                coef(p, 0, AUX.data(), a, b, c);
                for (int l = 0; l < nl; ++l)
                {
                    const Real inv = Real(1) / b[l];
                    cp[l] = c[l] * inv;
                    d[l] *= inv;
                }
                for (int g = 1; g < m; ++g)
                {
                    coef(p, g, AUX.data() + static_cast<std::size_t>(g) * nl, a, b, c);
                    Real *__restrict dg = d + static_cast<std::size_t>(g) * nl;
                    const Real *__restrict dm = dg - nl;
                    Real *__restrict cg = cp + static_cast<std::size_t>(g) * nl;
                    const Real *__restrict cm = cg - nl;
#pragma omp simd
                    for (int l = 0; l < nl; ++l)
                    {
                        const Real inv = Real(1) / (b[l] - a[l] * cm[l]);
                        cg[l] = c[l] * inv;
                        dg[l] = (dg[l] - a[l] * dm[l]) * inv;
                    }
                }
                for (int g = m - 2; g >= 0; --g)
                {
                    Real *__restrict dg = d + static_cast<std::size_t>(g) * nl;
                    const Real *__restrict dp = dg + nl;
                    const Real *__restrict cg = cp + static_cast<std::size_t>(g) * nl;
#pragma omp simd
                    for (int l = 0; l < nl; ++l)
                        dg[l] -= cg[l] * dp[l];
                }
                scatter(p, D.data());
            }
        }
    }

    // ------------------------------------------------------------------ uniform, local
    template <class Gather, class Scatter>
    void uniform_local(int np, int nl, const UniformMatrix &M, Gather &gather, Scatter &scatter)
    {
        const int m = M.m;
        const std::size_t ps = static_cast<std::size_t>(m) * nl;
        const Real *__restrict inv = M.inv.data();
        const Real *__restrict cp = M.cp.data();
        const Real *__restrict a = M.a_.data();
#pragma omp parallel
        {
            std::vector<Real> D(ps), AUX(ps);
#pragma omp for schedule(static)
            for (int p = 0; p < np; ++p)
            {
                gather(p, D.data(), AUX.data());
                Real *__restrict d = D.data();
                {
                    const Real s = inv[0];
#pragma omp simd
                    for (int l = 0; l < nl; ++l)
                        d[l] *= s;
                }
                for (int g = 1; g < m; ++g)
                {
                    Real *__restrict dg = d + static_cast<std::size_t>(g) * nl;
                    const Real *__restrict dm = dg - nl;
                    const Real ag = a[g], ig = inv[g];
#pragma omp simd
                    for (int l = 0; l < nl; ++l)
                        dg[l] = (dg[l] - ag * dm[l]) * ig;
                }
                for (int g = m - 2; g >= 0; --g)
                {
                    Real *__restrict dg = d + static_cast<std::size_t>(g) * nl;
                    const Real *__restrict dp = dg + nl;
                    const Real cg = cp[g];
#pragma omp simd
                    for (int l = 0; l < nl; ++l)
                        dg[l] -= cg * dp[l];
                }
                scatter(p, D.data());
            }
        }
    }

    // ------------------------------------------------------------------ general, distributed
    template <class Gather, class Coef, class Scatter>
    void sweep_distributed(const Decomposition &dec, int d, int np, int m, int nl,
                           Gather &gather, Coef &coef, Scatter &scatter)
    {
#ifdef USE_MPI
        const std::size_t ps = static_cast<std::size_t>(m) * nl;
        const std::size_t nlines = static_cast<std::size_t>(np) * nl;
        W_.resize(ps * np);
        WA_.resize(ps * np);
        WC_.resize(ps * np);
        send_.resize(6 * nlines);

        // ---- Phase 1: local modified Thomas elimination (no communication)
#pragma omp parallel
        {
            std::vector<Real> A(nl), B(nl), C(nl), AUX(ps);
#pragma omp for schedule(static)
            for (int p = 0; p < np; ++p)
            {
                Real *__restrict D = W_.data() + ps * p;
                Real *__restrict AA = WA_.data() + ps * p;
                Real *__restrict CC = WC_.data() + ps * p;
                gather(p, D, AUX.data());
                Real *a = A.data(), *b = B.data(), *c = C.data();
                // rows 0 and 1: normalise
                for (int g = 0; g < 2; ++g)
                {
                    coef(p, g, AUX.data() + static_cast<std::size_t>(g) * nl, a, b, c);
                    Real *dg = D + g * nl, *ag = AA + g * nl, *cg = CC + g * nl;
                    for (int l = 0; l < nl; ++l)
                    {
                        const Real inv = Real(1) / b[l];
                        ag[l] = a[l] * inv;
                        cg[l] = c[l] * inv;
                        dg[l] *= inv;
                    }
                }
                // forward: express row g in terms of x_0, x_g, x_{g+1}
                for (int g = 2; g < m; ++g)
                {
                    coef(p, g, AUX.data() + static_cast<std::size_t>(g) * nl, a, b, c);
                    Real *__restrict dg = D + static_cast<std::size_t>(g) * nl;
                    Real *__restrict ag = AA + static_cast<std::size_t>(g) * nl;
                    Real *__restrict cg = CC + static_cast<std::size_t>(g) * nl;
                    const Real *__restrict dm = dg - nl;
                    const Real *__restrict am = ag - nl;
                    const Real *__restrict cm = cg - nl;
#pragma omp simd
                    for (int l = 0; l < nl; ++l)
                    {
                        const Real inv = Real(1) / (b[l] - a[l] * cm[l]);
                        dg[l] = (dg[l] - a[l] * dm[l]) * inv;
                        ag[l] = -a[l] * am[l] * inv;
                        cg[l] = c[l] * inv;
                    }
                }
                // backward: express row g in terms of x_0, x_g, x_{m-1}
                for (int g = m - 3; g >= 1; --g)
                {
                    Real *__restrict dg = D + static_cast<std::size_t>(g) * nl;
                    Real *__restrict ag = AA + static_cast<std::size_t>(g) * nl;
                    Real *__restrict cg = CC + static_cast<std::size_t>(g) * nl;
                    const Real *__restrict dp = dg + nl;
                    const Real *__restrict ap = ag + nl;
                    const Real *__restrict cpp = cg + nl;
#pragma omp simd
                    for (int l = 0; l < nl; ++l)
                    {
                        dg[l] -= cg[l] * dp[l];
                        ag[l] -= cg[l] * ap[l];
                        cg[l] = -cg[l] * cpp[l];
                    }
                }
                // row 0: eliminate x_1 (not needed when m = 2: x_1 is the last unknown)
                if (m > 2)
                {
                    Real *d0 = D, *a0 = AA, *c0 = CC;
                    const Real *d1 = D + nl, *a1 = AA + nl, *c1 = CC + nl;
                    for (int l = 0; l < nl; ++l)
                    {
                        const Real inv = Real(1) / (Real(1) - c0[l] * a1[l]);
                        d0[l] = (d0[l] - c0[l] * d1[l]) * inv;
                        a0[l] = a0[l] * inv;
                        c0[l] = -c0[l] * c1[l] * inv;
                    }
                }
                // pack the two interface rows of every lane
                const Real *dL = D + static_cast<std::size_t>(m - 1) * nl;
                const Real *aL = AA + static_cast<std::size_t>(m - 1) * nl;
                const Real *cL = CC + static_cast<std::size_t>(m - 1) * nl;
                Real *s = send_.data() + 6 * static_cast<std::size_t>(p) * nl;
                for (int l = 0; l < nl; ++l)
                {
                    s[6 * l + 0] = AA[l];
                    s[6 * l + 1] = CC[l];
                    s[6 * l + 2] = D[l];
                    s[6 * l + 3] = aL[l];
                    s[6 * l + 4] = cL[l];
                    s[6 * l + 5] = dL[l];
                }
            }
        }

        solve_reduced(dec, d, nlines);

        // ---- Phase 4: recover the interior unknowns and scatter
#pragma omp parallel for schedule(static)
        for (int p = 0; p < np; ++p)
        {
            Real *__restrict D = W_.data() + ps * p;
            const Real *__restrict AA = WA_.data() + ps * p;
            const Real *__restrict CC = WC_.data() + ps * p;
            const Real *__restrict xf = xfl_.data() + 2 * static_cast<std::size_t>(p) * nl;
            for (int g = 1; g < m - 1; ++g)
            {
                Real *__restrict dg = D + static_cast<std::size_t>(g) * nl;
                const Real *__restrict ag = AA + static_cast<std::size_t>(g) * nl;
                const Real *__restrict cg = CC + static_cast<std::size_t>(g) * nl;
                for (int l = 0; l < nl; ++l)
                    dg[l] -= ag[l] * xf[2 * l] + cg[l] * xf[2 * l + 1];
            }
            Real *d0 = D, *dL = D + static_cast<std::size_t>(m - 1) * nl;
            for (int l = 0; l < nl; ++l)
            {
                d0[l] = xf[2 * l];
                dL[l] = xf[2 * l + 1];
            }
            scatter(p, D);
        }
#else
        (void)dec;
        (void)d;
        sweep_local(np, m, nl, gather, coef, scatter);
#endif
    }

    // ------------------------------------------------------------------ uniform, distributed
    template <class Gather, class Scatter>
    void uniform_distributed(const Decomposition &dec, int d, int np, int nl, const UniformMatrix &M,
                             Gather &gather, Scatter &scatter)
    {
#ifdef USE_MPI
        const int m = M.m;
        const std::size_t ps = static_cast<std::size_t>(m) * nl;
        const std::size_t nlines = static_cast<std::size_t>(np) * nl;
        W_.resize(ps * np);
        send_.resize(6 * nlines);
        const Real *__restrict inv = M.inv.data();
        const Real *__restrict a = M.a_.data();
        const Real *__restrict bC = M.bC.data();
        const Real *__restrict A = M.A.data();
        const Real *__restrict C = M.C.data();

#pragma omp parallel
        {
            std::vector<Real> AUX(ps);
#pragma omp for schedule(static)
            for (int p = 0; p < np; ++p)
            {
                Real *__restrict D = W_.data() + ps * p;
                gather(p, D, AUX.data());
                for (int g = 0; g < 2; ++g)
                {
                    Real *__restrict dg = D + static_cast<std::size_t>(g) * nl;
                    const Real s = inv[g];
#pragma omp simd
                    for (int l = 0; l < nl; ++l)
                        dg[l] *= s;
                }
                for (int g = 2; g < m; ++g)
                {
                    Real *__restrict dg = D + static_cast<std::size_t>(g) * nl;
                    const Real *__restrict dm = dg - nl;
                    const Real ag = a[g], ig = inv[g];
#pragma omp simd
                    for (int l = 0; l < nl; ++l)
                        dg[l] = (dg[l] - ag * dm[l]) * ig;
                }
                for (int g = m - 3; g >= 1; --g)
                {
                    Real *__restrict dg = D + static_cast<std::size_t>(g) * nl;
                    const Real *__restrict dp = dg + nl;
                    const Real cg = bC[g];
#pragma omp simd
                    for (int l = 0; l < nl; ++l)
                        dg[l] -= cg * dp[l];
                }
                {
                    Real *__restrict d0 = D;
                    const Real *__restrict d1 = D + nl;
                    const Real c0 = M.r0C, i0 = M.r0inv;
#pragma omp simd
                    for (int l = 0; l < nl; ++l)
                        d0[l] = (d0[l] - c0 * d1[l]) * i0;
                }
                const Real *dL = D + static_cast<std::size_t>(m - 1) * nl;
                Real *s = send_.data() + 6 * static_cast<std::size_t>(p) * nl;
                for (int l = 0; l < nl; ++l)
                {
                    s[6 * l + 0] = A[0];
                    s[6 * l + 1] = C[0];
                    s[6 * l + 2] = D[l];
                    s[6 * l + 3] = A[m - 1];
                    s[6 * l + 4] = C[m - 1];
                    s[6 * l + 5] = dL[l];
                }
            }
        }

        solve_reduced(dec, d, nlines);

#pragma omp parallel for schedule(static)
        for (int p = 0; p < np; ++p)
        {
            Real *__restrict D = W_.data() + ps * p;
            const Real *__restrict xf = xfl_.data() + 2 * static_cast<std::size_t>(p) * nl;
            for (int g = 1; g < m - 1; ++g)
            {
                Real *__restrict dg = D + static_cast<std::size_t>(g) * nl;
                const Real ag = A[g], cg = C[g];
                for (int l = 0; l < nl; ++l)
                    dg[l] -= ag * xf[2 * l] + cg * xf[2 * l + 1];
            }
            Real *d0 = D, *dL = D + static_cast<std::size_t>(m - 1) * nl;
            for (int l = 0; l < nl; ++l)
            {
                d0[l] = xf[2 * l];
                dL[l] = xf[2 * l + 1];
            }
            scatter(p, D);
        }
#else
        (void)dec;
        (void)d;
        (void)np;
        (void)nl;
        (void)M;
        (void)gather;
        (void)scatter;
#endif
    }

#ifdef USE_MPI
    /// Phases 2-3: gather the interface rows of all the processes of the line communicator
    /// (one collective for the whole sweep) and solve the 2P x 2P reduced systems.
    void solve_reduced(const Decomposition &dec, int d, std::size_t nlines)
    {
        const int P = dec.P(d);
        const int r = dec.topo().coord(d);
        recv_.resize(6 * nlines * P);
        xfl_.resize(2 * nlines);
        MPI_Allgather(send_.data(), static_cast<int>(6 * nlines), mpi_real_type(),
                      recv_.data(), static_cast<int>(6 * nlines), mpi_real_type(), dec.topo().line(d));
        const int R = 2 * P;
#pragma omp parallel
        {
            std::vector<Real> lo(R), up(R), rh(R);
#pragma omp for schedule(static)
            for (long L = 0; L < static_cast<long>(nlines); ++L)
            {
                for (int q = 0; q < P; ++q)
                {
                    const Real *v = recv_.data() + 6 * (static_cast<std::size_t>(q) * nlines + L);
                    lo[2 * q] = v[0];
                    up[2 * q] = v[1];
                    rh[2 * q] = v[2];
                    lo[2 * q + 1] = v[3];
                    up[2 * q + 1] = v[4];
                    rh[2 * q + 1] = v[5];
                }
                // Thomas on the reduced system (unit diagonal)
                for (int i = 1; i < R; ++i)
                {
                    const Real inv = Real(1) / (Real(1) - lo[i] * up[i - 1]);
                    up[i] *= inv;
                    rh[i] = (rh[i] - lo[i] * rh[i - 1]) * inv;
                }
                for (int i = R - 2; i >= 0; --i)
                    rh[i] -= up[i] * rh[i + 1];
                xfl_[2 * L] = rh[2 * r];
                xfl_[2 * L + 1] = rh[2 * r + 1];
            }
        }
    }
#else
    void solve_reduced(const Decomposition &, int, std::size_t) {}
#endif
};

#endif // TRIDIAGONAL_SOLVER_HPP
