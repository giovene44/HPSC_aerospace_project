#ifndef FAST_EXPRESSION_HPP
#define FAST_EXPRESSION_HPP

/**
 * @file Expression.hpp
 * @brief Small, fast compiler/evaluator for the analytic expressions read from the input
 *        files (forcing terms, exact solution, boundary data, permeability).
 *
 * The expressions are parsed once into a DAG (common sub-expressions are shared and
 * constant sub-trees are folded). Every node knows on which of the variables x, y, z, t it
 * depends, which allows two evaluation modes:
 *
 *  - operator()(x, y, z, t): plain point-wise evaluation;
 *  - eval_block(...):        evaluation on a tensor-product block of points.  Sub-expressions
 *                            that do not depend on the "line" variable are evaluated once per
 *                            line, sub-expressions that depend only on the line variable are
 *                            tabulated once per call, the remaining ones are evaluated with
 *                            simple (vectorisable) loops over the line.
 *
 * For the typical manufactured solutions (products of sin/cos of single variables) the block
 * evaluation costs a few nanoseconds per point, i.e. two to three orders of magnitude less
 * than evaluating the same expression with muparserx point by point.
 *
 * Supported syntax (muparserx compatible for everything used in this project):
 *   numbers (1, 2.5, 1e-3, .5), variables x y z t, constants pi e,
 *   + - * / ^ (right associative), unary + -, parentheses,
 *   sin cos tan asin acos atan sinh cosh tanh asinh acosh atanh
 *   exp log(=ln) ln log10 log2 sqrt cbrt abs sign floor ceil,
 *   pow(a,b) atan2(a,b) min(a,b) max(a,b) hypot(a,b)
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cctype>

namespace fastexpr
{

enum class Op : std::uint8_t
{
    Const,
    VarX,
    VarY,
    VarZ,
    VarT,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    PowInt, // power with a small integer exponent (stored in value)
    Neg,
    Sin,
    Cos,
    Tan,
    ASin,
    ACos,
    ATan,
    SinH,
    CosH,
    TanH,
    ASinH,
    ACosH,
    ATanH,
    Exp,
    Log,
    Log10,
    Log2,
    Sqrt,
    Cbrt,
    Abs,
    Sign,
    Floor,
    Ceil,
    ATan2,
    Min,
    Max,
    Hypot
};

enum : unsigned
{
    DEP_X = 1u,
    DEP_Y = 2u,
    DEP_Z = 4u,
    DEP_T = 8u
};

struct Node
{
    Op op = Op::Const;
    int a = -1;
    int b = -1;
    double value = 0.0;
    unsigned deps = 0;
};

inline bool is_binary(Op op)
{
    switch (op)
    {
    case Op::Add:
    case Op::Sub:
    case Op::Mul:
    case Op::Div:
    case Op::Pow:
    case Op::ATan2:
    case Op::Min:
    case Op::Max:
    case Op::Hypot:
        return true;
    default:
        return false;
    }
}

inline double ipow(double x, int n)
{
    bool inv = n < 0;
    unsigned m = static_cast<unsigned>(inv ? -n : n);
    double r = 1.0, p = x;
    while (m)
    {
        if (m & 1u)
            r *= p;
        p *= p;
        m >>= 1u;
    }
    return inv ? 1.0 / r : r;
}

inline double apply_unary(Op op, double x, double extra)
{
    switch (op)
    {
    case Op::Neg:
        return -x;
    case Op::PowInt:
        return ipow(x, static_cast<int>(extra));
    case Op::Sin:
        return std::sin(x);
    case Op::Cos:
        return std::cos(x);
    case Op::Tan:
        return std::tan(x);
    case Op::ASin:
        return std::asin(x);
    case Op::ACos:
        return std::acos(x);
    case Op::ATan:
        return std::atan(x);
    case Op::SinH:
        return std::sinh(x);
    case Op::CosH:
        return std::cosh(x);
    case Op::TanH:
        return std::tanh(x);
    case Op::ASinH:
        return std::asinh(x);
    case Op::ACosH:
        return std::acosh(x);
    case Op::ATanH:
        return std::atanh(x);
    case Op::Exp:
        return std::exp(x);
    case Op::Log:
        return std::log(x);
    case Op::Log10:
        return std::log10(x);
    case Op::Log2:
        return std::log2(x);
    case Op::Sqrt:
        return std::sqrt(x);
    case Op::Cbrt:
        return std::cbrt(x);
    case Op::Abs:
        return std::fabs(x);
    case Op::Sign:
        return (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0);
    case Op::Floor:
        return std::floor(x);
    case Op::Ceil:
        return std::ceil(x);
    default:
        throw std::logic_error("fastexpr: not a unary operator");
    }
}

inline double apply_binary(Op op, double x, double y)
{
    switch (op)
    {
    case Op::Add:
        return x + y;
    case Op::Sub:
        return x - y;
    case Op::Mul:
        return x * y;
    case Op::Div:
        return x / y;
    case Op::Pow:
        return std::pow(x, y);
    case Op::ATan2:
        return std::atan2(x, y);
    case Op::Min:
        return std::min(x, y);
    case Op::Max:
        return std::max(x, y);
    case Op::Hypot:
        return std::hypot(x, y);
    default:
        throw std::logic_error("fastexpr: not a binary operator");
    }
}

class Expression
{
public:
    /// Default: the constant zero.
    Expression() { root_ = add_const(0.0); }

    explicit Expression(const std::string &source) : source_(source)
    {
        Parser p(source, *this);
        root_ = p.parse();
        prune();
    }

    const std::string &source() const { return source_; }

    /// Variables the expression depends on (bitmask of DEP_X, DEP_Y, DEP_Z, DEP_T).
    unsigned dependencies() const { return nodes_[root_].deps; }
    bool depends_on_space() const { return (dependencies() & (DEP_X | DEP_Y | DEP_Z)) != 0; }
    bool is_constant() const { return dependencies() == 0; }

    /// Point-wise evaluation.
    double operator()(double x, double y, double z, double t) const
    {
        const std::size_t n = nodes_.size();
        double small[128];
        std::vector<double> big;
        double *v = small;
        if (n > 128)
        {
            big.resize(n);
            v = big.data();
        }
        for (std::size_t i = 0; i < n; ++i)
        {
            const Node &nd = nodes_[i];
            switch (nd.op)
            {
            case Op::Const:
                v[i] = nd.value;
                break;
            case Op::VarX:
                v[i] = x;
                break;
            case Op::VarY:
                v[i] = y;
                break;
            case Op::VarZ:
                v[i] = z;
                break;
            case Op::VarT:
                v[i] = t;
                break;
            default:
                v[i] = is_binary(nd.op) ? apply_binary(nd.op, v[nd.a], v[nd.b])
                                        : apply_unary(nd.op, v[nd.a], nd.value);
            }
        }
        return v[root_];
    }

    /**
     * @brief Evaluate on the tensor-product block {(xs[i], ys[j], zs[k])}, writing
     *        out[i*sx + j*sy + k*sz]. The longest of the three directions is used as
     *        "line" direction for the vectorised evaluation.
     */
    void eval_block(const double *xs, int nx, const double *ys, int ny, const double *zs, int nz,
                    double t, double *out, long sx, long sy, long sz) const
    {
        if (nx <= 0 || ny <= 0 || nz <= 0)
            return;
        // choose the line axis
        int axis = 0;
        if (ny > nx && ny >= nz)
            axis = 1;
        else if (nz > nx && nz > ny)
            axis = 2;

        const double *c[3] = {xs, ys, zs};
        const int n[3] = {nx, ny, nz};
        const long s[3] = {sx, sy, sz};
        const int A = axis, B = (axis + 1) % 3, C = (axis + 2) % 3;
        const unsigned depA = 1u << A;
        // Directions with a single point are constants for the whole call. The remaining
        // nodes are evaluated once per call (level 0), once per outer index kc (level 1:
        // depend on C only) or once per line (level 2: depend on B).
        const unsigned bitB = (n[B] > 1) ? (1u << B) : 0u;
        const unsigned bitC = (n[C] > 1) ? (1u << C) : 0u;

        const std::size_t nn = nodes_.size();
        const int L = n[A];

        std::vector<double> sval(nn, 0.0); // scalar values
        std::vector<int> arr(nn, -1);      // index of the array buffer, -1 => scalar
        std::vector<int> lev1, lev2;
        int narr = 0;
        for (std::size_t i = 0; i < nn; ++i)
        {
            const unsigned d = nodes_[i].deps;
            if (d & depA)
                arr[i] = narr++;
            if (d & bitB)
                lev2.push_back(static_cast<int>(i));
            else if (d & bitC)
                lev1.push_back(static_cast<int>(i));
        }
        std::vector<double> buf(static_cast<std::size_t>(narr) * static_cast<std::size_t>(L));
        auto A_ptr = [&](int node) -> double *
        { return buf.data() + static_cast<std::size_t>(arr[node]) * L; };
        auto eval_node = [&](std::size_t i, const double *p)
        {
            if (arr[i] < 0)
                sval[i] = eval_scalar_node(i, sval, p, t);
            else
                eval_array_node(i, sval, arr, buf, L, c[A], A, t);
        };

        double p[3] = {0.0, 0.0, 0.0};
        p[B] = c[B][0];
        p[C] = c[C][0];
        for (std::size_t i = 0; i < nn; ++i) // level 0
            if (!(nodes_[i].deps & (bitB | bitC)))
                eval_node(i, p);

        for (int kc = 0; kc < n[C]; ++kc)
        {
            p[C] = c[C][kc];
            for (int i : lev1)
                eval_node(i, p);
            for (int kb = 0; kb < n[B]; ++kb)
            {
                p[B] = c[B][kb];
                for (int i : lev2)
                    eval_node(i, p);
                double *o = out + kb * s[B] + kc * s[C];
                const long sa = s[A];
                if (arr[root_] < 0)
                {
                    const double val = sval[root_];
                    for (int ia = 0; ia < L; ++ia)
                        o[ia * sa] = val;
                }
                else
                {
                    const double *r = A_ptr(root_);
                    if (sa == 1)
                        std::memcpy(o, r, sizeof(double) * static_cast<std::size_t>(L));
                    else
                        for (int ia = 0; ia < L; ++ia)
                            o[ia * sa] = r[ia];
                }
            }
        }
    }

private:
    std::vector<Node> nodes_;
    int root_ = 0;
    std::string source_;
    std::unordered_map<std::string, int> cache_; // hash-consing for CSE

    // ------------------------------------------------------------------ building
    /// Remove the nodes not reachable from the root (left behind by the simplifications),
    /// keeping the topological order.
    void prune()
    {
        const int n = static_cast<int>(nodes_.size());
        std::vector<char> live(n, 0);
        live[root_] = 1;
        for (int i = n - 1; i >= 0; --i)
        {
            if (!live[i])
                continue;
            if (nodes_[i].a >= 0)
                live[nodes_[i].a] = 1;
            if (nodes_[i].b >= 0)
                live[nodes_[i].b] = 1;
        }
        std::vector<int> map(n, -1);
        std::vector<Node> kept;
        for (int i = 0; i < n; ++i)
            if (live[i])
            {
                Node nd = nodes_[i];
                if (nd.a >= 0)
                    nd.a = map[nd.a];
                if (nd.b >= 0)
                    nd.b = map[nd.b];
                map[i] = static_cast<int>(kept.size());
                kept.push_back(nd);
            }
        nodes_.swap(kept);
        root_ = map[root_];
        cache_.clear(); // the expression is complete: no more hash-consing needed
    }

    static std::string key_of(const Node &n)
    {
        std::string k(1, static_cast<char>(n.op));
        k.append(reinterpret_cast<const char *>(&n.a), sizeof(int));
        k.append(reinterpret_cast<const char *>(&n.b), sizeof(int));
        k.append(reinterpret_cast<const char *>(&n.value), sizeof(double));
        return k;
    }

    int push(Node n)
    {
        const std::string k = key_of(n);
        auto it = cache_.find(k);
        if (it != cache_.end())
            return it->second;
        nodes_.push_back(n);
        const int id = static_cast<int>(nodes_.size()) - 1;
        cache_.emplace(k, id);
        return id;
    }

    int add_const(double v)
    {
        Node n;
        n.op = Op::Const;
        n.value = v;
        return push(n);
    }

    int add_var(Op op, unsigned dep)
    {
        Node n;
        n.op = op;
        n.deps = dep;
        return push(n);
    }

    bool is_const(int id, double &v) const
    {
        if (nodes_[id].op == Op::Const)
        {
            v = nodes_[id].value;
            return true;
        }
        return false;
    }

    int add_unary(Op op, int a, double extra = 0.0)
    {
        double va;
        if (is_const(a, va))
            return add_const(apply_unary(op, va, extra));
        if (op == Op::Neg && nodes_[a].op == Op::Neg)
            return nodes_[a].a;
        Node n;
        n.op = op;
        n.a = a;
        n.value = extra;
        n.deps = nodes_[a].deps;
        return push(n);
    }

    int add_binary(Op op, int a, int b)
    {
        double va, vb;
        const bool ca = is_const(a, va), cb = is_const(b, vb);
        if (ca && cb)
            return add_const(apply_binary(op, va, vb));
        // algebraic simplifications that are exact in floating point
        if (op == Op::Mul)
        {
            if ((ca && va == 1.0))
                return b;
            if ((cb && vb == 1.0))
                return a;
        }
        if (op == Op::Add)
        {
            if (ca && va == 0.0)
                return b;
            if (cb && vb == 0.0)
                return a;
        }
        if (op == Op::Sub && cb && vb == 0.0)
            return a;
        // A*X +- B*X -> (A +- B)*X when X is the same node: products are built with the
        // factors depending on x last (see Parser::product), so sums of terms with the same
        // x-dependent factor need one multiplication per point instead of one per term.
        if ((op == Op::Add || op == Op::Sub) && nodes_[a].op == Op::Mul && nodes_[b].op == Op::Mul &&
            nodes_[a].b == nodes_[b].b && (nodes_[nodes_[a].b].deps & ~(nodes_[nodes_[a].a].deps | nodes_[nodes_[b].a].deps)))
        {
            const int x = nodes_[a].b;
            return add_binary(Op::Mul, add_binary(op, nodes_[a].a, nodes_[b].a), x);
        }
        if (op == Op::Div && cb && vb == 1.0)
            return a;
        if (op == Op::Pow && cb && vb == std::floor(vb) && std::fabs(vb) <= 64.0)
        {
            if (vb == 1.0)
                return a;
            return add_unary(Op::PowInt, a, vb);
        }
        Node n;
        n.op = op;
        n.a = a;
        n.b = b;
        n.deps = nodes_[a].deps | nodes_[b].deps;
        return push(n);
    }

    // ------------------------------------------------------------------ evaluation helpers
    double eval_scalar_node(std::size_t i, const std::vector<double> &sval, const double *p, double t) const
    {
        const Node &nd = nodes_[i];
        switch (nd.op)
        {
        case Op::Const:
            return nd.value;
        case Op::VarX:
            return p[0];
        case Op::VarY:
            return p[1];
        case Op::VarZ:
            return p[2];
        case Op::VarT:
            return t;
        default:
            return is_binary(nd.op) ? apply_binary(nd.op, sval[nd.a], sval[nd.b])
                                    : apply_unary(nd.op, sval[nd.a], nd.value);
        }
    }

    void eval_array_node(std::size_t i, const std::vector<double> &sval, const std::vector<int> &arr,
                         std::vector<double> &buf, int L, const double *coord, int axis, double t) const
    {
        (void)t;
        const Node &nd = nodes_[i];
        double *o = buf.data() + static_cast<std::size_t>(arr[i]) * L;
        switch (nd.op)
        {
        case Op::VarX:
        case Op::VarY:
        case Op::VarZ:
            (void)axis; // the only array variable is the line variable
            std::memcpy(o, coord, sizeof(double) * static_cast<std::size_t>(L));
            return;
        default:
            break;
        }
        const double *pa = (arr[nd.a] >= 0) ? buf.data() + static_cast<std::size_t>(arr[nd.a]) * L : nullptr;
        const double sa = (arr[nd.a] >= 0) ? 0.0 : sval[nd.a];
        if (!is_binary(nd.op))
        {
            // pa is never null here: an array node with a scalar child would be scalar
            switch (nd.op)
            {
            case Op::Neg:
                for (int k = 0; k < L; ++k)
                    o[k] = -pa[k];
                return;
            case Op::PowInt:
            {
                const int e = static_cast<int>(nd.value);
                if (e == 2)
                    for (int k = 0; k < L; ++k)
                        o[k] = pa[k] * pa[k];
                else if (e == 3)
                    for (int k = 0; k < L; ++k)
                        o[k] = pa[k] * pa[k] * pa[k];
                else
                    for (int k = 0; k < L; ++k)
                        o[k] = ipow(pa[k], e);
                return;
            }
            case Op::Sin:
                for (int k = 0; k < L; ++k)
                    o[k] = std::sin(pa[k]);
                return;
            case Op::Cos:
                for (int k = 0; k < L; ++k)
                    o[k] = std::cos(pa[k]);
                return;
            case Op::Exp:
                for (int k = 0; k < L; ++k)
                    o[k] = std::exp(pa[k]);
                return;
            default:
                for (int k = 0; k < L; ++k)
                    o[k] = apply_unary(nd.op, pa[k], nd.value);
                return;
            }
        }
        const double *pb = (arr[nd.b] >= 0) ? buf.data() + static_cast<std::size_t>(arr[nd.b]) * L : nullptr;
        const double sb = (arr[nd.b] >= 0) ? 0.0 : sval[nd.b];
#define FASTEXPR_BIN(EXPR)                       \
    if (pa && pb)                                \
        for (int k = 0; k < L; ++k)              \
        {                                        \
            const double xa = pa[k], xb = pb[k]; \
            o[k] = (EXPR);                       \
        }                                        \
    else if (pa)                                 \
        for (int k = 0; k < L; ++k)              \
        {                                        \
            const double xa = pa[k], xb = sb;    \
            o[k] = (EXPR);                       \
        }                                        \
    else                                         \
        for (int k = 0; k < L; ++k)              \
        {                                        \
            const double xa = sa, xb = pb[k];    \
            o[k] = (EXPR);                       \
        }
        switch (nd.op)
        {
        case Op::Add:
            FASTEXPR_BIN(xa + xb);
            return;
        case Op::Sub:
            FASTEXPR_BIN(xa - xb);
            return;
        case Op::Mul:
            FASTEXPR_BIN(xa * xb);
            return;
        case Op::Div:
            FASTEXPR_BIN(xa / xb);
            return;
        default:
            FASTEXPR_BIN(apply_binary(nd.op, xa, xb));
            return;
        }
#undef FASTEXPR_BIN
    }

    // ------------------------------------------------------------------ parser
    class Parser
    {
    public:
        Parser(const std::string &s, Expression &e) : s_(s), e_(e) {}

        int parse()
        {
            int r = parse_expr();
            skip();
            if (pos_ != s_.size())
                error("unexpected character '" + std::string(1, s_[pos_]) + "'");
            return r;
        }

    private:
        const std::string &s_;
        Expression &e_;
        std::size_t pos_ = 0;

        [[noreturn]] void error(const std::string &msg) const
        {
            throw std::runtime_error("Expression parse error: " + msg + " at position " +
                                     std::to_string(pos_) + " in \"" + s_ + "\"");
        }

        void skip()
        {
            while (pos_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[pos_])))
                ++pos_;
        }

        bool accept(char c)
        {
            skip();
            if (pos_ < s_.size() && s_[pos_] == c)
            {
                ++pos_;
                return true;
            }
            return false;
        }

        void expect(char c)
        {
            if (!accept(c))
                error(std::string("expected '") + c + "'");
        }

        int parse_expr()
        {
            int l = parse_term();
            for (;;)
            {
                if (accept('+'))
                    l = e_.add_binary(Op::Add, l, parse_term());
                else if (accept('-'))
                    l = e_.add_binary(Op::Sub, l, parse_term());
                else
                    return l;
            }
        }

        // A product chain f1 * f2 / f3 * ... is rebuilt as (prod of numerators) / (prod of
        // denominators), with the factors sorted by the variables they depend on (constants,
        // then t, z, y and x last). With left-associative products the partial products
        // then depend on as few variables as possible, so that eval_block computes most of
        // them once per line instead of once per point, e.g.
        //   cos(t)*sin(x)*sin(y)*sin(z) -> ((cos(t)*sin(z))*sin(y))*sin(x).
        int parse_term()
        {
            std::vector<int> num{parse_unary()}, den;
            for (;;)
            {
                skip();
                if (pos_ < s_.size() && s_[pos_] == '*' && !(pos_ + 1 < s_.size() && s_[pos_ + 1] == '*'))
                {
                    ++pos_;
                    num.push_back(parse_unary());
                }
                else if (accept('/'))
                    den.push_back(parse_unary());
                else
                    break;
            }
            const int n = product(num);
            if (den.empty())
                return n;
            return e_.add_binary(Op::Div, n, product(den));
        }

        int product(std::vector<int> f)
        {
            auto key = [this](int id)
            {
                const unsigned d = e_.nodes_[id].deps;
                return ((d & DEP_X) ? 8 : 0) + ((d & DEP_Y) ? 4 : 0) + ((d & DEP_Z) ? 2 : 0) + ((d & DEP_T) ? 1 : 0);
            };
            std::stable_sort(f.begin(), f.end(), [&](int a, int b) { return key(a) < key(b); });
            int r = f[0];
            for (std::size_t i = 1; i < f.size(); ++i)
                r = e_.add_binary(Op::Mul, r, f[i]);
            return r;
        }

        int parse_unary()
        {
            if (accept('-'))
                return e_.add_unary(Op::Neg, parse_unary());
            if (accept('+'))
                return parse_unary();
            return parse_power();
        }

        int parse_power()
        {
            int base = parse_primary();
            skip();
            if (accept('^'))
                return e_.add_binary(Op::Pow, base, parse_unary());
            if (pos_ + 1 < s_.size() && s_[pos_] == '*' && s_[pos_ + 1] == '*')
            {
                pos_ += 2;
                return e_.add_binary(Op::Pow, base, parse_unary());
            }
            return base;
        }

        int parse_primary()
        {
            skip();
            if (pos_ >= s_.size())
                error("unexpected end of expression");
            const char ch = s_[pos_];
            if (ch == '(')
            {
                ++pos_;
                int r = parse_expr();
                expect(')');
                return r;
            }
            if (std::isdigit(static_cast<unsigned char>(ch)) || ch == '.')
                return parse_number();
            if (std::isalpha(static_cast<unsigned char>(ch)) || ch == '_')
            {
                std::size_t b = pos_;
                while (pos_ < s_.size() && (std::isalnum(static_cast<unsigned char>(s_[pos_])) || s_[pos_] == '_'))
                    ++pos_;
                const std::string id = s_.substr(b, pos_ - b);
                skip();
                if (pos_ < s_.size() && s_[pos_] == '(')
                    return parse_call(id);
                if (id == "x")
                    return e_.add_var(Op::VarX, DEP_X);
                if (id == "y")
                    return e_.add_var(Op::VarY, DEP_Y);
                if (id == "z")
                    return e_.add_var(Op::VarZ, DEP_Z);
                if (id == "t")
                    return e_.add_var(Op::VarT, DEP_T);
                if (id == "pi" || id == "Pi" || id == "PI")
                    return e_.add_const(M_PI);
                if (id == "e")
                    return e_.add_const(M_E);
                error("unknown identifier '" + id + "'");
            }
            error(std::string("unexpected character '") + ch + "'");
        }

        int parse_number()
        {
            const char *begin = s_.c_str() + pos_;
            char *end = nullptr;
            const double v = std::strtod(begin, &end);
            if (end == begin)
                error("invalid number");
            pos_ += static_cast<std::size_t>(end - begin);
            return e_.add_const(v);
        }

        int parse_call(const std::string &id)
        {
            expect('(');
            std::vector<int> args;
            if (!accept(')'))
            {
                do
                {
                    args.push_back(parse_expr());
                } while (accept(','));
                expect(')');
            }
            static const std::unordered_map<std::string, Op> unary = {
                {"sin", Op::Sin}, {"cos", Op::Cos}, {"tan", Op::Tan}, {"asin", Op::ASin}, {"acos", Op::ACos}, {"atan", Op::ATan}, {"sinh", Op::SinH}, {"cosh", Op::CosH}, {"tanh", Op::TanH}, {"asinh", Op::ASinH}, {"acosh", Op::ACosH}, {"atanh", Op::ATanH}, {"exp", Op::Exp}, {"log", Op::Log}, {"ln", Op::Log}, {"log10", Op::Log10}, {"log2", Op::Log2}, {"sqrt", Op::Sqrt}, {"cbrt", Op::Cbrt}, {"abs", Op::Abs}, {"sign", Op::Sign}, {"floor", Op::Floor}, {"ceil", Op::Ceil}};
            static const std::unordered_map<std::string, Op> binary = {
                {"pow", Op::Pow}, {"atan2", Op::ATan2}, {"min", Op::Min}, {"max", Op::Max}, {"hypot", Op::Hypot}};
            auto u = unary.find(id);
            if (u != unary.end())
            {
                if (args.size() != 1)
                    error("function '" + id + "' expects 1 argument");
                return e_.add_unary(u->second, args[0]);
            }
            auto b = binary.find(id);
            if (b != binary.end())
            {
                if (id == "min" || id == "max")
                {
                    if (args.empty())
                        error("function '" + id + "' expects at least 1 argument");
                    int r = args[0];
                    for (std::size_t i = 1; i < args.size(); ++i)
                        r = e_.add_binary(b->second, r, args[i]);
                    return r;
                }
                if (args.size() != 2)
                    error("function '" + id + "' expects 2 arguments");
                return e_.add_binary(b->second, args[0], args[1]);
            }
            error("unknown function '" + id + "'");
        }
    };
};

} // namespace fastexpr

#endif // FAST_EXPRESSION_HPP
