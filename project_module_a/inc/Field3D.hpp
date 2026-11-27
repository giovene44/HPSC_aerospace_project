#ifndef FIELD3D_HPP
#define FIELD3D_HPP

#include "Variables.hpp"
#include <vector>
#include <cstddef>
#include <algorithm>

/**
 * @brief Local block of a 3D field with one layer of ghost cells on every side.
 *
 * Owned (local) indices are i in [0,nx), j in [0,ny), k in [0,nz); ghost cells are
 * at -1 and n. Storage is x-fastest: offset = (i+1) + sy*(j+1) + sz*(k+1).
 */
class Field3D
{
public:
    Field3D() = default;
    Field3D(int nx_, int ny_, int nz_) { resize(nx_, ny_, nz_); }

    void resize(int nx_, int ny_, int nz_)
    {
        nx = nx_;
        ny = ny_;
        nz = nz_;
        sy = nx + 2;
        sz = sy * (ny + 2);
        data.assign(static_cast<std::size_t>(sz) * static_cast<std::size_t>(nz + 2), Real(0));
    }

    long idx(int i, int j, int k) const { return (i + 1) + sy * (j + 1) + sz * (k + 1); }
    Real &operator()(int i, int j, int k) { return data[idx(i, j, k)]; }
    Real operator()(int i, int j, int k) const { return data[idx(i, j, k)]; }

    Real *raw() { return data.data(); }
    const Real *raw() const { return data.data(); }

    /// Stride of a unit step along direction d (0=x, 1=y, 2=z).
    long stride(int d) const { return d == 0 ? 1 : (d == 1 ? sy : sz); }
    int n(int d) const { return d == 0 ? nx : (d == 1 ? ny : nz); }

    void fill(Real v) { std::fill(data.begin(), data.end(), v); }

    int nx = 0, ny = 0, nz = 0;
    long sy = 0, sz = 0;
    std::vector<Real> data;
};

#endif // FIELD3D_HPP
