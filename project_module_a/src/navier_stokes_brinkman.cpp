#include "navier_stokes_brinkman.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <utility>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{
enum RowType : int
{
    ROW_STD = 0,      // interior row:           -w, 1 + 2w, -w
    ROW_DIRICHLET,    // node on the boundary:    0, 1, 0
    ROW_CLOSE_LO,     // boundary h/2 to the left of the first node (ghost extrapolation)
    ROW_CLOSE_HI      // boundary h/2 to the right of the last node (ghost extrapolation)
};
constexpr Real C83 = Real(8) / Real(3);
constexpr Real C43 = Real(4) / Real(3);
constexpr Real C13 = Real(1) / Real(3);
} // namespace

// =====================================================================================
// Construction / initialisation
// =====================================================================================

NavierStokesBrinkman::NavierStokesBrinkman(const ProblemData &data, const Decomposition &dec, Real dt, Real t0)
    : dec_(dec), dt_(dt), t_(t0), nu_(data.nu), chi_(data.chi), L_(data.L)
{
    for (int d = 0; d < 3; ++d)
    {
        N_[d] = dec.N(d);
        n_[d] = dec.n(d);
        g0_[d] = dec.begin(d);
        h_[d] = L_[d] / (Real(N_[d]) - Real(0.5));
    }
    for (int c = 0; c < 3; ++c)
    {
        f_[c] = fastexpr::Expression(data.forcing[c]);
        uex_[c] = fastexpr::Expression(data.u_exact[c]);
        ubc_[c] = fastexpr::Expression(data.u_boundary[c]);
    }
    pex_ = fastexpr::Expression(data.p_exact);
    k_ = fastexpr::Expression(data.k_expr);

    // node coordinates
    for (int d = 0; d < 3; ++d)
    {
        xp_[d].resize(n_[d]);
        for (int i = 0; i < n_[d]; ++i)
            xp_[d][i] = Real(g0_[d] + i) * h_[d];
        for (int c = 0; c < 3; ++c)
        {
            xc_[c][d].resize(n_[d]);
            const Real shift = (c == d) ? Real(0.5) : Real(0);
            for (int i = 0; i < n_[d]; ++i)
                xc_[c][d][i] = (Real(g0_[d] + i) + shift) * h_[d];
        }
    }

    for (int c = 0; c < 3; ++c)
    {
        u_[c].resize(n_[0], n_[1], n_[2]);
        eta_[c].resize(n_[0], n_[1], n_[2]);
        zeta_[c].resize(n_[0], n_[1], n_[2]);
        cs_[c].resize(n_[0], n_[1], n_[2]);
        cu_[c].resize(n_[0], n_[1], n_[2]);
    }
    p_.resize(n_[0], n_[1], n_[2]);
#ifdef _OPENMP
    fbuf_.resize(omp_get_max_threads());
#else
    fbuf_.resize(1);
#endif
    if (chi_ != Real(0))
    {
        div_new_.resize(n_[0], n_[1], n_[2]);
        div_old_.resize(n_[0], n_[1], n_[2]);
    }
    phi_.resize(n_[0], n_[1], n_[2]);
    pstar_.resize(n_[0], n_[1], n_[2]);
    psi_.resize(n_[0], n_[1], n_[2]);

    // boundary faces
    for (int c = 0; c < 3; ++c)
        for (int e = 0; e < 3; ++e)
        {
            const int a = (e == 0) ? 1 : 0, b = (e == 2) ? 1 : 2;
            for (auto *F : {&bon_[c][e], &bhalf_old_[c][e], &bhalf_new_[c][e]})
            {
                F->a = a;
                F->b = b;
                F->na = n_[a];
                F->nb = n_[b];
            }
            bon_[c][e].active = (e != c) ? dec.at_lo(e) : dec.at_hi(e);
            bhalf_old_[c][e].active = bhalf_new_[c][e].active = (e != c) ? dec.at_hi(e) : dec.at_lo(e);
            for (auto *F : {&bon_[c][e], &bhalf_old_[c][e], &bhalf_new_[c][e]})
                if (F->active)
                    F->v.assign(static_cast<std::size_t>(F->na) * F->nb, Real(0));
        }

    // Brinkman coefficients at the staggered positions of every component:
    //   beta = nu dt / (2k),  cs = dt / (1 + beta),  cu = (1 - beta) / (1 + beta)
    for (int c = 0; c < 3; ++c)
    {
        eval_field(k_, c, Real(0), cs_[c]);
        for (int k = 0; k < n_[2]; ++k)
            for (int j = 0; j < n_[1]; ++j)
                for (int i = 0; i < n_[0]; ++i)
                    if (!(cs_[c](i, j, k) > 0) || !std::isfinite(cs_[c](i, j, k)))
                        throw std::runtime_error("NavierStokesBrinkman: the permeability k must be positive and finite");
        Real *CS = cs_[c].raw();
        Real *CU = cu_[c].raw();
        const std::size_t tot = cs_[c].data.size();
        for (std::size_t q = 0; q < tot; ++q)
        {
            const Real kval = CS[q];
            // ghost entries are never used: give them harmless values
            const Real beta = (kval > Real(0) && std::isfinite(kval)) ? nu_ * dt_ / (Real(2) * kval) : Real(0);
            CS[q] = dt_ / (Real(1) + beta);
            CU[q] = (Real(1) - beta) / (Real(1) + beta);
        }
    }

    // Matrices shared by all the lines of a sweep: factorised once.
    //  - pressure: (I - Dxx) with Neumann rows, always;
    //  - velocity: (I - gamma Dxx) when k is constant (gamma is then the same everywhere).
    uniform_v_ = k_.is_constant();
    for (int d = 0; d < 3; ++d)
    {
        const int m = n_[d];
        const Real w = Real(1) / (h_[d] * h_[d]);
        std::vector<Real> a(m, -w), b(m, Real(1) + Real(2) * w), cc(m, -w);
        for (int g = 0; g < m; ++g)
        {
            const int G = g0_[d] + g;
            if (G == 0) // node on the boundary, symmetric ghost: dphi/dn = 0
            {
                a[g] = Real(0);
                cc[g] = -Real(2) * w;
            }
            else if (G == N_[d] - 1) // boundary half a cell beyond: phi_N = phi_{N-1}
            {
                b[g] = Real(1) + w;
                cc[g] = Real(0);
            }
        }
        ump_[d].set(dec_, d, a, b, cc);
        if (!uniform_v_)
            continue;
        for (int c = 0; c < 3; ++c)
        {
            const Real wv = Real(0.5) * nu_ * cs_[c](0, 0, 0) * w;
            std::vector<Real> av(m, -wv), bv(m, Real(1) + Real(2) * wv), cv(m, -wv);
            for (int g = 0; g < m; ++g)
            {
                const int rt = row_type(c, d, g0_[d] + g);
                if (rt == ROW_DIRICHLET)
                {
                    av[g] = cv[g] = Real(0);
                    bv[g] = Real(1);
                }
                else if (rt == ROW_CLOSE_LO)
                {
                    av[g] = Real(0);
                    bv[g] = Real(1) + Real(4) * wv;
                    cv[g] = -C43 * wv;
                }
                else if (rt == ROW_CLOSE_HI)
                {
                    av[g] = -C43 * wv;
                    bv[g] = Real(1) + Real(4) * wv;
                    cv[g] = Real(0);
                }
            }
            umv_[d][c].set(dec_, d, av, bv, cv);
        }
    }
}

int NavierStokesBrinkman::row_type(int c, int d, int G) const
{
    if (d != c)
    {
        if (G == 0)
            return ROW_DIRICHLET;
        if (G == N_[d] - 1)
            return ROW_CLOSE_HI;
    }
    else
    {
        if (G == N_[d] - 1)
            return ROW_DIRICHLET;
        if (G == 0)
            return ROW_CLOSE_LO;
    }
    return ROW_STD;
}

void NavierStokesBrinkman::eval_field(const fastexpr::Expression &ex, int comp, Real t, Field3D &F) const
{
    const std::array<std::vector<Real>, 3> &X = (comp < 3) ? xc_[comp] : xp_;
    const int nx = n_[0], ny = n_[1], nz = n_[2];
#pragma omp parallel for schedule(static)
    for (int k = 0; k < nz; ++k)
        ex.eval_block(X[0].data(), nx, X[1].data(), ny, X[2].data() + k, 1, t,
                      F.raw() + F.idx(0, 0, k), 1, F.sy, F.sz);
}

void NavierStokesBrinkman::eval_face(const fastexpr::Expression &ex, int c, int e, Real Xe, Real t, Face &F) const
{
    const Real *coords[3];
    int nn[3];
    long st[3];
    for (int d = 0; d < 3; ++d)
    {
        if (d == e)
        {
            coords[d] = &Xe;
            nn[d] = 1;
            st[d] = 0;
        }
        else
        {
            coords[d] = xc_[c][d].data();
            nn[d] = n_[d];
            st[d] = (d == F.a) ? 1 : F.na;
        }
    }
    ex.eval_block(coords[0], nn[0], coords[1], nn[1], coords[2], nn[2], t, F.v.data(), st[0], st[1], st[2]);
}

void NavierStokesBrinkman::update_faces(Real t, std::array<std::array<Face, 3>, 3> &half, bool on_node)
{
    for (int c = 0; c < 3; ++c)
        for (int e = 0; e < 3; ++e)
        {
            if (on_node && bon_[c][e].active)
                eval_face(ubc_[c], c, e, (e != c) ? Real(0) : L_[e], t, bon_[c][e]);
            if (half[c][e].active)
                eval_face(ubc_[c], c, e, (e != c) ? L_[e] : Real(0), t, half[c][e]);
        }
}

void NavierStokesBrinkman::fill_physical_ghost(int c, int d, Field3D &F,
                                               const std::array<std::array<Face, 3>, 3> &half) const
{
    const Face &fc = half[c][d];
    if (!fc.active)
        return;
    const long sd = F.stride(d);
    const int gl = (d != c) ? n_[d] : -1;       // ghost layer
    const int g0 = (d != c) ? n_[d] - 1 : 0;     // first node next to the boundary
    const long s1 = (d != c) ? -sd : sd;         // towards the interior
    const int a = fc.a, b = fc.b;
    for (int ib = 0; ib < n_[b]; ++ib)
        for (int ia = 0; ia < n_[a]; ++ia)
        {
            int idx[3];
            idx[a] = ia;
            idx[b] = ib;
            idx[d] = g0;
            const long q = F.idx(idx[0], idx[1], idx[2]);
            idx[d] = gl;
            const long qg = F.idx(idx[0], idx[1], idx[2]);
            const Real bval = fc.v[ia + static_cast<std::size_t>(fc.na) * ib];
            F.data[qg] = C83 * bval - Real(2) * F.data[q] + C13 * F.data[q + s1];
        }
}

void NavierStokesBrinkman::initialize()
{
    for (int c = 0; c < 3; ++c)
        eval_field(uex_[c], c, t_, u_[c]);
    // Dirichlet nodes take the boundary data
    update_faces(t_, bhalf_old_, true);
    for (int c = 0; c < 3; ++c)
        for (int e = 0; e < 3; ++e)
        {
            const Face &F = bon_[c][e];
            if (!F.active)
                continue;
            const int g = (e != c) ? 0 : n_[e] - 1;
            for (int ib = 0; ib < n_[F.b]; ++ib)
                for (int ia = 0; ia < n_[F.a]; ++ia)
                {
                    int idx[3];
                    idx[F.a] = ia;
                    idx[F.b] = ib;
                    idx[e] = g;
                    u_[c](idx[0], idx[1], idx[2]) = F.v[ia + static_cast<std::size_t>(F.na) * ib];
                }
        }
    for (int c = 0; c < 3; ++c)
    {
        eta_[c].data = u_[c].data;
        zeta_[c].data = u_[c].data;
    }
    // pressure is staggered in time: p^{-1/2}, and phi^{-1/2} = p(t0 + dt/2) - p(t0 - dt/2)
    // so that the first predictor p* is exact.
    eval_field(pex_, 3, t_ - Real(0.5) * dt_, p_);
    eval_field(pex_, 3, t_ + Real(0.5) * dt_, phi_);
    for (std::size_t q = 0; q < p_.data.size(); ++q)
        phi_.data[q] -= p_.data[q];

    // ghosts used by the divergence (errors() may be called before the first step)
    for (int c = 0; c < 3; ++c)
    {
        dec_.exchange(u_[c], c);
        fill_physical_ghost(c, c, u_[c], bhalf_old_);
    }
    if (chi_ != Real(0))
        for (int k = 0; k < n_[2]; ++k)
            for (int j = 0; j < n_[1]; ++j)
                for (int i = 0; i < n_[0]; ++i)
                    div_old_(i, j, k) = (u_[0](i, j, k) - u_[0](i - 1, j, k)) / h_[0] +
                                    (u_[1](i, j, k) - u_[1](i, j - 1, k)) / h_[1] +
                                    (u_[2](i, j, k) - u_[2](i, j, k - 1)) / h_[2];
}

// =====================================================================================
// Line sweeps
// =====================================================================================

template <int D>
void NavierStokesBrinkman::velocity_sweep(int c)
{
    constexpr int LA = (D == 0) ? 1 : 0; // lane direction
    constexpr int PL = (D == 2) ? 1 : 2; // plane direction
    Field3D &F = (D == 0) ? eta_[c] : (D == 1 ? zeta_[c] : u_[c]);
    const Field3D &S = (D == 1) ? eta_[c] : zeta_[c]; // source of the y and z sweeps
    const Real *__restrict CS = cs_[c].raw();
    const Real *__restrict CU = cu_[c].raw();
    Real *__restrict FV = F.raw();
    const Real *__restrict SV = S.raw();

    const int m = n_[D], nl = n_[LA], np = n_[PL];
    const long sR = F.stride(D), sL = F.stride(LA), sP = F.stride(PL);
    const long base0 = F.idx(0, 0, 0);
    const Real ih2 = Real(1) / (h_[D] * h_[D]);
    const Real hnu = Real(0.5) * nu_; // gamma = hnu * cs
    const Real nu = nu_;
    const bool uniform = uniform_v_;

    // data of the explicit step (only D == 0)
    const Real *__restrict ETA = eta_[c].raw();
    const Real *__restrict ZETA = zeta_[c].raw();
    const Real *__restrict U = u_[c].raw();
    const Real *__restrict PS = pstar_.raw();
    const long sy = F.sy, sz = F.sz, sc = F.stride(c);
    const Real ihx2 = Real(1) / (h_[0] * h_[0]), ihy2 = Real(1) / (h_[1] * h_[1]), ihz2 = Real(1) / (h_[2] * h_[2]);
    const Real ihc = Real(1) / h_[c];

    const Face &fD = bon_[c][D];
    const Face &fL = bon_[c][LA];
    const Face &fP = bon_[c][PL];
    const Face &fhn = bhalf_new_[c][D];
    const Face &fho = bhalf_old_[c][D];

    // only the first and the last local rows can be boundary rows
    const int rt_first = row_type(c, D, g0_[D]);
    const int rt_last = row_type(c, D, g0_[D] + m - 1);
    std::vector<char> known_lane(nl), known_plane(np);
    for (int l = 0; l < nl; ++l)
        known_lane[l] = on_node_boundary(c, LA, g0_[LA] + l) ? 1 : 0;
    for (int p = 0; p < np; ++p)
        known_plane[p] = on_node_boundary(c, PL, g0_[PL] + p) ? 1 : 0;

    auto face_at = [](const Face &f, int p, int g, int l) -> Real
    {
        int idx[3];
        idx[D] = g;
        idx[LA] = l;
        idx[PL] = p;
        return f.at(idx[0], idx[1], idx[2]);
    };
    // right-hand side of the delta form at node q (interior rows); fo = forcing at q
    auto rhs_at = [&](long q, Real cs, Real cu, Real fo) -> Real
    {
        if constexpr (D == 0)
        {
            // xi - eta^n with xi = cu u + cs (f - dp/dx_c + nu (Dxx eta + Dyy zeta + Dzz u))
            const Real lap = (ETA[q + 1] - Real(2) * ETA[q] + ETA[q - 1]) * ihx2 +
                             (ZETA[q + sy] - Real(2) * ZETA[q] + ZETA[q - sy]) * ihy2 +
                             (U[q + sz] - Real(2) * U[q] + U[q - sz]) * ihz2;
            const Real grad = (PS[q + sc] - PS[q]) * ihc;
            return cu * U[q] + cs * (fo - grad + nu * lap) - ETA[q];
        }
        else
        {
            (void)cs;
            (void)cu;
            (void)fo;
            return SV[q] - FV[q];
        }
    };
    const Real cs0 = cs_[c](0, 0, 0), cu0 = cu_[c](0, 0, 0); // used when k is constant
    // boundary rows of the line (p, l): Dirichlet node or ghost closure
    auto fix_row = [&](int p, int l, int g, int rt, long q, Real &val)
    {
        if (rt == ROW_DIRICHLET)
            val = face_at(fD, p, g, l) - FV[q];
        else if (rt == ROW_CLOSE_LO || rt == ROW_CLOSE_HI)
            // boundary-data increment of the ghost closure: + gamma 8/(3h^2) (b^{n+1} - b^n)
            val += hnu * CS[q] * ih2 * C83 * (face_at(fhn, p, g, l) - face_at(fho, p, g, l));
    };

    // Lines lying on a Dirichlet face (known lanes / planes) are solved with a zero rhs and
    // then overwritten with the boundary data in scatter.
    auto gather = [&](int p, Real *__restrict Dp, Real *__restrict AUX)
    {
        const bool pk = known_plane[p] != 0;
        if constexpr (D == 0)
        {
            // forcing f^{n+1/2} on this plane (z = const), evaluated here so that it never
            // goes through main memory
            Real *__restrict FP = nullptr;
            if (!pk)
            {
#ifdef _OPENMP
                std::vector<Real> &fb = fbuf_[omp_get_thread_num()];
#else
                std::vector<Real> &fb = fbuf_[0];
#endif
                fb.resize(static_cast<std::size_t>(m) * nl);
                FP = fb.data();
                f_[c].eval_block(xc_[c][0].data(), m, xc_[c][1].data(), nl, xc_[c][2].data() + p, 1, th_, FP, 1, m, 0);
            }
            // rows along x are contiguous in memory: loop over the lanes, then along the line
            for (int l = 0; l < nl; ++l)
            {
                const long lb = base0 + p * sP + l * sL;
                if (pk || known_lane[l])
                {
                    for (int g = 0; g < m; ++g)
                        Dp[static_cast<std::size_t>(g) * nl + l] = Real(0);
                    continue;
                }
                const Real *__restrict fl = FP + static_cast<std::size_t>(l) * m;
                if (uniform)
                {
#pragma omp simd
                    for (int g = 0; g < m; ++g)
                        Dp[static_cast<std::size_t>(g) * nl + l] = rhs_at(lb + g, cs0, cu0, fl[g]);
                }
                else
                {
#pragma omp simd
                    for (int g = 0; g < m; ++g)
                        Dp[static_cast<std::size_t>(g) * nl + l] = rhs_at(lb + g, CS[lb + g], CU[lb + g], fl[g]);
                }
                if (!uniform)
                {
#pragma omp simd
                    for (int g = 0; g < m; ++g)
                        AUX[static_cast<std::size_t>(g) * nl + l] = hnu * CS[lb + g] * ih2;
                }
                if (rt_first != ROW_STD)
                    fix_row(p, l, 0, rt_first, lb, Dp[l]);
                if (rt_last != ROW_STD)
                    fix_row(p, l, m - 1, rt_last, lb + (m - 1), Dp[static_cast<std::size_t>(m - 1) * nl + l]);
            }
        }
        else
        {
            (void)AUX;
            for (int g = 0; g < m; ++g)
            {
                Real *__restrict row = Dp + static_cast<std::size_t>(g) * nl;
                if (pk)
                {
                    for (int l = 0; l < nl; ++l)
                        row[l] = Real(0);
                    continue;
                }
                const long rb = base0 + p * sP + g * sR;
#pragma omp simd
                for (int l = 0; l < nl; ++l)
                    row[l] = rhs_at(rb + l, Real(0), Real(0), Real(0));
                const int rt = (g == 0) ? rt_first : (g == m - 1 ? rt_last : ROW_STD);
                if (rt != ROW_STD)
                    for (int l = 0; l < nl; ++l)
                        fix_row(p, l, g, rt, rb + l, row[l]);
                for (int l = 0; l < nl; ++l)
                    if (known_lane[l])
                        row[l] = Real(0);
            }
        }
    };

    auto scatter = [&](int p, const Real *__restrict Dp)
    {
        const bool pk = known_plane[p] != 0;
        if constexpr (D == 0)
        {
            for (int l = 0; l < nl; ++l)
            {
                const long lb = base0 + p * sP + l * sL;
                if (pk || known_lane[l])
                {
                    const Face &fk = pk ? fP : fL;
                    for (int g = 0; g < m; ++g)
                        FV[lb + g] = face_at(fk, p, g, l);
                    continue;
                }
#pragma omp simd
                for (int g = 0; g < m; ++g)
                    FV[lb + g] += Dp[static_cast<std::size_t>(g) * nl + l];
            }
        }
        else
        {
            for (int g = 0; g < m; ++g)
            {
                const Real *__restrict row = Dp + static_cast<std::size_t>(g) * nl;
                const long rb = base0 + p * sP + g * sR;
                if (pk)
                {
                    for (int l = 0; l < nl; ++l)
                        FV[rb + l] = face_at(fP, p, g, l);
                    continue;
                }
#pragma omp simd
                for (int l = 0; l < nl; ++l)
                    FV[rb + l] += row[l];
                for (int l = 0; l < nl; ++l)
                    if (known_lane[l])
                        FV[rb + l] = face_at(fL, p, g, l);
            }
        }
    };

    if (uniform)
    {
        solver_.sweep_uniform(dec_, D, np, nl, umv_[D][c], gather, scatter);
        return;
    }

    auto coef = [&](int p, int g, const Real *__restrict aux, Real *__restrict A, Real *__restrict B,
                    Real *__restrict C)
    {
        const int rt = (g == 0) ? rt_first : (g == m - 1 ? rt_last : ROW_STD);
        if (rt == ROW_DIRICHLET)
        {
            for (int l = 0; l < nl; ++l)
            {
                A[l] = Real(0);
                B[l] = Real(1);
                C[l] = Real(0);
            }
            return;
        }
        const long rb = base0 + p * sP + g * sR;
        const Real ka = (rt == ROW_STD) ? Real(1) : (rt == ROW_CLOSE_LO ? Real(0) : C43);
        const Real kb = (rt == ROW_STD) ? Real(2) : Real(4);
        const Real kc = (rt == ROW_STD) ? Real(1) : (rt == ROW_CLOSE_LO ? C43 : Real(0));
#pragma omp simd
        for (int l = 0; l < nl; ++l)
        {
            const Real w = (D == 0) ? aux[l] : hnu * CS[rb + l] * ih2;
            A[l] = -ka * w;
            B[l] = Real(1) + kb * w;
            C[l] = -kc * w;
        }
    };
    solver_.sweep(dec_, D, np, m, nl, gather, coef, scatter);
}

template <int D>
void NavierStokesBrinkman::pressure_sweep(const Field3D *src, Field3D &dst)
{
    constexpr int LA = (D == 0) ? 1 : 0;
    constexpr int PL = (D == 2) ? 1 : 2;
    const int m = n_[D], nl = n_[LA], np = n_[PL];
    const long sR = dst.stride(D), sL = dst.stride(LA), sP = dst.stride(PL);
    const long base0 = dst.idx(0, 0, 0);
    const Real *__restrict SRC = src ? src->raw() : nullptr;
    Real *__restrict DST = dst.raw();
    const Real *__restrict U0 = u_[0].raw();
    const Real *__restrict U1 = u_[1].raw();
    const Real *__restrict U2 = u_[2].raw();
    const long sy = dst.sy, sz = dst.sz;
    const Real ihx = Real(1) / h_[0], ihy = Real(1) / h_[1], ihz = Real(1) / h_[2];
    const Real idt = Real(1) / dt_;
    Real *__restrict DIVN = (chi_ != Real(0)) ? div_new_.raw() : nullptr;

    auto gather = [&](int p, Real *__restrict Dp, Real *)
    {
        if constexpr (D == 0)
        {
            // first sweep: rhs = -div(u^{n+1}) / dt, rows along x (contiguous)
            (void)SRC;
            const int Gk = g0_[PL] + p;
            for (int l = 0; l < nl; ++l)
            {
                const long lb = base0 + p * sP + l * sL;
                Real *__restrict col = Dp + l;
#pragma omp simd
                for (int g = 0; g < m; ++g)
                {
                    const long q = lb + g;
                    const Real div = (U0[q] - U0[q - 1]) * ihx + (U1[q] - U1[q - sy]) * ihy + (U2[q] - U2[q - sz]) * ihz;
                    col[static_cast<std::size_t>(g) * nl] = -div * idt;
                }
                if (DIVN)
                    for (int g = 0; g < m; ++g)
                        DIVN[lb + g] = -col[static_cast<std::size_t>(g) * nl] * dt_;
                // Pressure nodes on the edges x=0/y=0, x=0/z=0, y=0/z=0 are not seen by any
                // velocity unknown (all their neighbouring velocities are Dirichlet nodes), so
                // their divergence, fixed by the boundary data, cannot be corrected: using it
                // as a source would make the pressure drift by O(h^2)/dt at every step.
                const int zr = (g0_[LA] + l == 0) + (Gk == 0);
                if (zr == 2)
                {
                    for (int g = 0; g < m; ++g)
                        col[static_cast<std::size_t>(g) * nl] = Real(0);
                }
                else if (zr == 1 && g0_[D] == 0)
                    col[0] = Real(0);
            }
        }
        else
        {
            for (int g = 0; g < m; ++g)
            {
                Real *__restrict row = Dp + static_cast<std::size_t>(g) * nl;
                const long rb = base0 + p * sP + g * sR;
#pragma omp simd
                for (int l = 0; l < nl; ++l)
                    row[l] = SRC[rb + l];
            }
        }
    };

    auto scatter = [&](int p, const Real *__restrict Dp)
    {
        if constexpr (D == 0)
        {
            for (int l = 0; l < nl; ++l)
            {
                const long lb = base0 + p * sP + l * sL;
#pragma omp simd
                for (int g = 0; g < m; ++g)
                    DST[lb + g] = Dp[static_cast<std::size_t>(g) * nl + l];
            }
        }
        else
        {
            for (int g = 0; g < m; ++g)
            {
                const Real *__restrict row = Dp + static_cast<std::size_t>(g) * nl;
                const long rb = base0 + p * sP + g * sR;
#pragma omp simd
                for (int l = 0; l < nl; ++l)
                    DST[rb + l] = row[l];
            }
        }
    };

    solver_.sweep_uniform(dec_, D, np, nl, ump_[D], gather, scatter);
}

// =====================================================================================
// Time stepping
// =====================================================================================

Field3D NavierStokesBrinkman::output_pressure() const
{
    // The pressure on the edges x=0/y=0, x=0/z=0, y=0/z=0 never influences the velocity
    // (see the pressure right-hand side): its value is defined here by second-order
    // extrapolation from the controlled nodes. The edges always lie on a process together
    // with the layer next to them (at least 2 local nodes per direction).
    Field3D P = p_;
    const bool z0[3] = {g0_[0] == 0, g0_[1] == 0, g0_[2] == 0};
    for (int d = 0; d < 3; ++d) // edge along d
    {
        const int a = (d + 1) % 3, b = (d + 2) % 3;
        if (!z0[a] || !z0[b])
            continue;
        for (int s = 0; s < n_[d]; ++s)
        {
            if (z0[d] && s == 0)
                continue; // corner, below
            int i0[3], ia[3], ib[3], iab[3];
            i0[d] = ia[d] = ib[d] = iab[d] = s;
            i0[a] = ib[a] = 0;
            ia[a] = iab[a] = 1;
            i0[b] = ia[b] = 0;
            ib[b] = iab[b] = 1;
            P(i0[0], i0[1], i0[2]) = P(ia[0], ia[1], ia[2]) + P(ib[0], ib[1], ib[2]) - P(iab[0], iab[1], iab[2]);
        }
    }
    if (z0[0] && z0[1] && z0[2])
        P(0, 0, 0) = P(1, 0, 0) + P(0, 1, 0) + P(0, 0, 1) - P(1, 1, 0) - P(1, 0, 1) - P(0, 1, 1) + P(1, 1, 1);
    return P;
}

Real NavierStokesBrinkman::weighted_mean(const Field3D &F) const
{
    const int nx = n_[0], ny = n_[1], nz = n_[2];
    const Real wx0 = (g0_[0] == 0) ? Real(0.5) : Real(1);
    const Real wy0 = (g0_[1] == 0) ? Real(0.5) : Real(1);
    const Real wz0 = (g0_[2] == 0) ? Real(0.5) : Real(1);
    double sum = 0;
#pragma omp parallel for reduction(+ : sum) schedule(static)
    for (int k = 0; k < nz; ++k)
    {
        double sk = 0;
        for (int j = 0; j < ny; ++j)
        {
            const Real *row = F.raw() + F.idx(0, j, k);
            double sj = wx0 * row[0];
            for (int i = 1; i < nx; ++i)
                sj += row[i];
            sk += (j == 0 ? wy0 : Real(1)) * sj;
        }
        sum += (k == 0 ? wz0 : Real(1)) * sk;
    }
    sum = dec_.topo().sum(sum);
    const double wtot = (double(N_[0]) - 0.5) * (double(N_[1]) - 0.5) * (double(N_[2]) - 0.5);
    return Real(sum / wtot);
}

void NavierStokesBrinkman::step()
{
    const Real t1 = t_ + dt_;
    const Real th = t_ + Real(0.5) * dt_;

    // pressure predictor p* = p^{n-1/2} + phi^{n-1/2}
    {
        const std::size_t tot = p_.data.size();
        Real *__restrict PS = pstar_.raw();
        const Real *__restrict P = p_.raw();
        const Real *__restrict PH = phi_.raw();
#pragma omp parallel for simd schedule(static)
        for (std::size_t q = 0; q < tot; ++q)
            PS[q] = P[q] + PH[q];
        for (int d = 0; d < 3; ++d)
            dec_.exchange(pstar_, d);
    }

    // boundary data at t^{n+1}, forcing at t^{n+1/2}
    update_faces(t1, bhalf_new_, true);
    th_ = th; // the forcing is evaluated inside the x sweeps

    // ghost layers for the explicit operator: Dxx eta^n, Dyy zeta^n, Dzz u^n
    for (int c = 0; c < 3; ++c)
    {
        dec_.exchange(eta_[c], 0);
        dec_.exchange(zeta_[c], 1);
        dec_.exchange(u_[c], 2);
        fill_physical_ghost(c, 0, eta_[c], bhalf_old_);
        fill_physical_ghost(c, 1, zeta_[c], bhalf_old_);
        fill_physical_ghost(c, 2, u_[c], bhalf_old_);
    }

    // momentum: three directional sweeps per component
    for (int c = 0; c < 3; ++c)
        velocity_sweep<0>(c);
    for (int c = 0; c < 3; ++c)
        velocity_sweep<1>(c);
    for (int c = 0; c < 3; ++c)
        velocity_sweep<2>(c);

    // divergence of u^{n+1}: ghost layers along the component direction
    for (int c = 0; c < 3; ++c)
    {
        dec_.exchange(u_[c], c);
        fill_physical_ghost(c, c, u_[c], bhalf_new_);
    }

    // pressure increment: (I - Dxx)(I - Dyy)(I - Dzz) phi = -div(u^{n+1}) / dt
    pressure_sweep<0>(nullptr, psi_);
    pressure_sweep<1>(&psi_, pstar_); // pstar_ is free now: used as work array
    pressure_sweep<2>(&pstar_, phi_);

    // phi on the uncontrolled edge nodes (see pressure_sweep) is meaningless: keep it zero
    // so that the pressure there stays bounded (it is never used by the velocity).
    for_each_uncontrolled([this](int i, int j, int k) { phi_(i, j, k) = Real(0); });

    // Pressure update. The constant is the only pressure mode not controlled by the momentum
    // equation (grad of a constant is zero) and it is an exact eigenvector, with eigenvalue 1,
    // of the factorised Neumann operator. Since the discrete boundary flux of the Dirichlet
    // data is zero only up to O(h^2), without a filter the pressure level would drift by
    // O(h^2)/dt at every step. Removing the mean of the increment (weights 1/2 on the boundary
    // nodes: the left eigenvector of the operator) changes nothing else.
    if (chi_ != Real(0))
    {
        // rotational form: p^{n+1/2} = p^{n-1/2} + phi - chi nu div(u^{n+1} + u^n) / 2
        const std::size_t tot = p_.data.size();
        const Real cr = Real(0.5) * chi_ * nu_;
        Real *__restrict PS = psi_.raw();
        const Real *__restrict PH = phi_.raw();
        const Real *__restrict DN = div_new_.raw();
        const Real *__restrict DO = div_old_.raw();
#pragma omp parallel for simd schedule(static)
        for (std::size_t q = 0; q < tot; ++q)
            PS[q] = PH[q] - cr * (DN[q] + DO[q]);
        for_each_uncontrolled([this](int i, int j, int k) { psi_(i, j, k) = Real(0); });
        const Real cm = weighted_mean(psi_);
        Real *__restrict P = p_.raw();
#pragma omp parallel for simd schedule(static)
        for (std::size_t q = 0; q < tot; ++q)
            P[q] += PS[q] - cm;
        std::swap(div_new_.data, div_old_.data);
    }
    else
    {
        const Real cmean = weighted_mean(phi_);
        const std::size_t tot = p_.data.size();
        Real *__restrict P = p_.raw();
        Real *__restrict PH = phi_.raw();
#pragma omp parallel for simd schedule(static)
        for (std::size_t q = 0; q < tot; ++q)
        {
            PH[q] -= cmean;
            P[q] += PH[q];
        }
    }

    t_ = t1;
    std::swap(bhalf_old_, bhalf_new_);
}

double NavierStokesBrinkman::advance(int nsteps)
{
    dec_.topo().barrier();
    const auto start = std::chrono::steady_clock::now();
    for (int n = 0; n < nsteps; ++n)
        step();
    const auto stop = std::chrono::steady_clock::now();
    const double local = std::chrono::duration<double>(stop - start).count();
    return dec_.topo().max(local);
}

// =====================================================================================
// Diagnostics
// =====================================================================================

ErrorNorms NavierStokesBrinkman::errors() const
{
    // Velocity: compared with u_ex(t^n). Pressure: compared with p_ex(t^n - dt/2) (the pressure
    // is staggered in time) modulo the kernel of the discrete gradient, i.e.
    //  - up to an additive constant (both are made zero-mean): with Dirichlet conditions on
    //    the whole boundary the pressure is defined up to a constant;
    //  - excluding the edge nodes that no velocity unknown sees (see pressure_sweep).
    const int nx = n_[0], ny = n_[1], nz = n_[2];
    const Real tp = t_ - Real(0.5) * dt_;
    std::vector<Real> pe(static_cast<std::size_t>(nx) * ny * nz), pex(pe.size());
    std::vector<char> ctrl(pe.size(), 1);
    for_each_uncontrolled([&](int i, int j, int k) { ctrl[i + static_cast<std::size_t>(nx) * (j + static_cast<std::size_t>(ny) * k)] = 0; });
    double eu = 0, nu2 = 0, dv = 0, mu = 0, se = 0, sx = 0, nctrl = 0;
#pragma omp parallel reduction(+ : eu, nu2, dv, se, sx, nctrl) reduction(max : mu)
    {
        std::vector<Real> buf(static_cast<std::size_t>(nx) * ny);
#pragma omp for schedule(static)
        for (int k = 0; k < nz; ++k)
        {
            for (int c = 0; c < 3; ++c)
            {
                uex_[c].eval_block(xc_[c][0].data(), nx, xc_[c][1].data(), ny, xc_[c][2].data() + k, 1, t_,
                                   buf.data(), 1, nx, 0);
                for (int j = 0; j < ny; ++j)
                    for (int i = 0; i < nx; ++i)
                    {
                        const Real ex = buf[i + static_cast<std::size_t>(nx) * j];
                        const Real e = u_[c](i, j, k) - ex;
                        eu += e * e;
                        nu2 += ex * ex;
                        mu = std::max(mu, double(std::fabs(e)));
                    }
            }
            Real *pk = pex.data() + static_cast<std::size_t>(nx) * ny * k;
            Real *ek = pe.data() + static_cast<std::size_t>(nx) * ny * k;
            pex_.eval_block(xp_[0].data(), nx, xp_[1].data(), ny, xp_[2].data() + k, 1, tp, pk, 1, nx, 0);
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i)
                {
                    const std::size_t q = i + static_cast<std::size_t>(nx) * j;
                    const bool on = ctrl[q + static_cast<std::size_t>(nx) * ny * k] != 0;
                    ek[q] = on ? p_(i, j, k) - pk[q] : Real(0);
                    if (on)
                    {
                        se += ek[q];
                        sx += pk[q];
                        ++nctrl;
                    }
                    const Real div = (u_[0](i, j, k) - u_[0](i - 1, j, k)) / h_[0] +
                                     (u_[1](i, j, k) - u_[1](i, j - 1, k)) / h_[1] +
                                     (u_[2](i, j, k) - u_[2](i, j, k - 1)) / h_[2];
                    dv += div * div;
                }
        }
    }
    double s1[6] = {eu, nu2, dv, se, sx, nctrl};
    dec_.topo().sum(s1, 6);
    const double emean = s1[3] / s1[5], xmean = s1[4] / s1[5];
    double ep = 0, np2 = 0, mp = 0;
#pragma omp parallel for reduction(+ : ep, np2) reduction(max : mp)
    for (std::size_t q = 0; q < pe.size(); ++q)
    {
        if (!ctrl[q])
            continue;
        const double e = pe[q] - emean, x = pex[q] - xmean;
        ep += e * e;
        np2 += x * x;
        mp = std::max(mp, std::fabs(e));
    }
    double s2[2] = {ep, np2};
    dec_.topo().sum(s2, 2);

    const double dV = double(h_[0]) * h_[1] * h_[2];
    ErrorNorms out;
    out.u_l2 = std::sqrt(s1[0] * dV);
    out.p_l2 = std::sqrt(s2[0] * dV);
    const double nu_ex = std::sqrt(s1[1] * dV), np_ex = std::sqrt(s2[1] * dV);
    out.u_l2_rel = nu_ex > 1e-300 ? out.u_l2 / nu_ex : out.u_l2;
    out.p_l2_rel = np_ex > 1e-300 ? out.p_l2 / np_ex : out.p_l2;
    out.div_l2 = std::sqrt(s1[2] * dV);
    out.u_linf = dec_.topo().max(mu);
    out.p_linf = dec_.topo().max(mp);
    return out;
}

void NavierStokesBrinkman::write_vtk(const std::string &prefix) const
{
    const int nx = n_[0], ny = n_[1], nz = n_[2];
    const std::string fname = prefix + "_rank" + std::to_string(dec_.topo().rank()) + ".vtk";
    std::ofstream f(fname);
    if (!f)
        throw std::runtime_error("Cannot open " + fname);
    f << "# vtk DataFile Version 3.0\nNavier-Stokes-Brinkman t=" << t_ << "\nASCII\nDATASET STRUCTURED_POINTS\n";
    f << "DIMENSIONS " << nx << " " << ny << " " << nz << "\n";
    f << "ORIGIN " << xp_[0][0] << " " << xp_[1][0] << " " << xp_[2][0] << "\n";
    f << "SPACING " << h_[0] << " " << h_[1] << " " << h_[2] << "\n";
    f << "POINT_DATA " << static_cast<long>(nx) * ny * nz << "\n";
    f << "SCALARS pressure double 1\nLOOKUP_TABLE default\n";
    const Field3D P = output_pressure();
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                f << P(i, j, k) << "\n";
    // velocity interpolated at the pressure nodes (one-sided on the lower boundary)
    f << "VECTORS velocity double\n";
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
            {
                const Real uu = Real(0.5) * (u_[0](i, j, k) + u_[0](i - 1, j, k));
                const Real vv = Real(0.5) * (u_[1](i, j, k) + u_[1](i, j - 1, k));
                const Real ww = Real(0.5) * (u_[2](i, j, k) + u_[2](i, j, k - 1));
                f << uu << " " << vv << " " << ww << "\n";
            }
}

// explicit instantiations
template void NavierStokesBrinkman::velocity_sweep<0>(int);
template void NavierStokesBrinkman::velocity_sweep<1>(int);
template void NavierStokesBrinkman::velocity_sweep<2>(int);
template void NavierStokesBrinkman::pressure_sweep<0>(const Field3D *, Field3D &);
template void NavierStokesBrinkman::pressure_sweep<1>(const Field3D *, Field3D &);
template void NavierStokesBrinkman::pressure_sweep<2>(const Field3D *, Field3D &);
