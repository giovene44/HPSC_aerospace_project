#ifndef VECTORVARIABLES_HPP
#define VECTORVARIABLES_HPP
#include "ScalarVariable.hpp"
#include "BoundaryFunctions.hpp"
#include <vector>
#include <iostream>

class VectorVariable
{
public:
    VectorVariable(Dim Nx_, Dim Ny_, Dim Nz_, Real dx_, Real dy_, Real dz_)
        : Nx(Nx_), Ny(Ny_), Nz(Nz_), dx(dx_), dy(dy_), dz(dz_)
    {
        data.clear();
        for (int a = 0; a < 3; ++a)
            data.push_back(ScalarVariable(Nx, Ny, Nz, dx, dy, dz));
    }

    inline Dim size() const noexcept
    {
        return data.size();
    }

    inline Dim elements_per_component() const noexcept
    {
        return Nx * Ny * Nz;
    }

    inline Dim get_Nx() const noexcept
    {
        return Nx;
    }

    inline Dim get_Ny() const noexcept
    {
        return Ny;
    }

    inline Dim get_Nz() const noexcept
    {
        return Nz;
    }

    // accessing method:
    // assuming i,j,k start from 0.
    inline Real value(int axes, Dim i, Dim j, Dim k) const
    {
        if (axes < 0 || axes >= 3)
            throw std::out_of_range("axes index out of range");
        if (i < 0 || i >= Nx)
            return 0.0;
        else if (j < 0 || j >= Ny)
            return 0.0;
        else if (k < 0 || k >= Nz)
            return 0.0;
        else
            return data[axes].get(i, j, k);
    }

    inline Real value(int axes, Dim index) const
    {
        if (axes < 0 || axes >= 3)
            throw std::out_of_range("axes index out of range");
        if (index < 0 || index >= Nx * Ny * Nz)
            throw std::out_of_range("index out of range");
        return data[axes].get(index);
    }

    /**
     * @brief Sets all elements in all three vector components (x, y, z) to the specified value.
     * This is implemented by calling set_all on the internal ScalarVariable objects.
     * @param value The Real value to assign to all elements.
     */
    void set_all(Real value)
    {
        for (int a = 0; a < static_cast<int>(data.size()); ++a)
        {
            data[a].set_all(value);
        }
    }

    void set_all(BoundaryFunctions &other, Real t)
    {

        for (Dim idx = 0; idx < Nx * Ny * Nz; ++idx)
        {
            Dim i = idx % Nx;
            Dim j = (idx / Nx) % Ny;
            Dim k = idx / (Nx * Ny);

            // Convert grid indices to physical coordinates
            Real x = i * dx;
            Real y = j * dy;
            Real z = k * dz;

            data[0].set(idx) = other.value<0>(x, y, z, t);
            data[1].set(idx) = other.value<1>(x, y, z, t);
            data[2].set(idx) = other.value<2>(x, y, z, t);
        }
    }

    inline Real &set(int axes, Dim i, Dim j, Dim k) noexcept
    {
        return data[axes].set(i, j, k);
    }

    inline Real &set(int axes, Dim index) noexcept
    {
        return data[axes].set(index);
    }

    inline ScalarVariable &set(const int axes) noexcept
    {
        return data[axes];
    }

    // -------------------------------------------------------------------------
    // FIRST DERIVATIVE (CENTERED DIFFERENCE - 2nd ORDER)
    // -------------------------------------------------------------------------
    // Uses centered difference (u_{i+1} - u_{i-1}) / (2*h) for interior points.
    // Uses one-sided difference at boundaries (1st Order).

    //TODO: This should be changed: it needs to shift to the pressure nodes!
    Real first_derivative(int axes, int derivation_direction, Dim i, Dim j, Dim k) const
    {

        Real v_plus = 0.0;
        Real v_minus = 0.0;
        Real den = 0.0;

        // we need to handle the x=0, y=0, z=0 borders outside of this method
        // because they require BC values for velocity!


        if (derivation_direction == 0)
        { // x
            if(i==0 && axes == 0)
                throw std::invalid_argument("VectorVariable::first_derivative: invalid i for centered difference");
        

            v_plus = value(axes, i, j, k);
            v_minus = value(axes, i-1, j, k);
            den = dx;

        }
        else if (derivation_direction == 1)
        { // y
            if(j==0 && axes == 1)
                throw std::invalid_argument("VectorVariable::first_derivative: invalid j for centered difference");

            v_plus = value(axes, i, j, k);
            v_minus = value(axes, i, j-1, k);
            den = dy;

        }
        else if (derivation_direction == 2)
        { // z
            if(k==0 && axes == 2)
                throw std::invalid_argument("VectorVariable::first_derivative: invalid k for centered difference");

            v_plus = value(axes, i, j, k);
            v_minus = value(axes, i, j, k-1);
            den = dz;

        }
        else
        {
            throw std::invalid_argument("VectorVariable::first_derivative: invalid derivation_direction");
        }

        return (v_plus - v_minus) / den;
    }

    Real first_derivative(int axes, int derivation_direction, Dim index) const
    {
        Dim i = index % Nx;
        Dim j = (index / Nx) % Ny;
        Dim k = index / (Nx * Ny);
        return first_derivative(axes, derivation_direction, i, j, k);
    }

    // second order derivative is computed on the velocity nodes!
    // on the borders we use a one-sided second order schema.
    // ENSURE Nx,Ny,Nz>=4 TO AVOID PROBLEMS!
    Real second_derivative(int axes, int derivation_direction, Dim i, Dim j, Dim k) const
    {
        Real v1(Real(0.0)), v2(Real(0.0)), v3(Real(0.0)), den(Real(0.0)), val(Real(0.0));

        if (derivation_direction == 0)
        { // x direction
            if (i == 0)
            {
                return (2.0 * value(axes, i, j, k) - 5.0 * value(axes, i + 1, j, k) + 4.0 * value(axes, i + 2, j, k) - value(axes, i + 3, j, k)) / (dx * dx);
            }
            else if (i == Nx - 1)
            {
                return (2.0 * value(axes, i, j, k) - 5.0 * value(axes, i - 1, j, k) + 4.0 * value(axes, i - 2, j, k) - value(axes, i - 3, j, k)) / (dx * dx);
            }
            v1 = value(axes, i + 1, j, k);
            v2 = value(axes, i, j, k);
            v3 = value(axes, i - 1, j, k);
            den = dx * dx;
        }
        else if (derivation_direction == 1)
        { // y direction
            if (j == 0)
            {
                return (2.0 * value(axes, i, j, k) - 5.0 * value(axes, i, j + 1, k) + 4.0 * value(axes, i, j + 2, k) - value(axes, i, j + 3, k)) / (dy * dy);
            }
            else if (j == Ny - 1)
            {
                return (2.0 * value(axes, i, j, k) - 5.0 * value(axes, i, j - 1, k) + 4.0 * value(axes, i, j - 2, k) - value(axes, i, j - 3, k)) / (dy * dy);
            }
            v1 = value(axes, i, j + 1, k);
            v2 = value(axes, i, j, k);
            v3 = value(axes, i, j - 1, k);
            den = dy * dy;
        }
        else if (derivation_direction == 2)
        { // z direction
            if (k == 0)
            {
                return (2.0 * value(axes, i, j, k) - 5.0 * value(axes, i, j, k + 1) + 4.0 * value(axes, i, j, k + 2) - value(axes, i, j, k + 3)) / (dz * dz);
            }
            else if (k == Nz - 1)
            {
                return (2.0 * value(axes, i, j, k) - 5.0 * value(axes, i, j, k - 1) + 4.0 * value(axes, i, j, k - 2) - value(axes, i, j, k - 3)) / (dz * dz);
            }
            v1 = value(axes, i, j, k + 1);
            v2 = value(axes, i, j, k);
            v3 = value(axes, i, j, k - 1);
            den = dz * dz;
        }
        else
        {
            throw std::invalid_argument("VectorVariable::second_derivative: invalid derivation_direction");
        }
        val = v1 - 2 * v2 + v3;
        val /= den;
        return val;
    }

    Real second_derivative(int axes, int derivation_direction, Dim index) const
    {
        Dim i = index % Nx;
        Dim j = (index / Nx) % Ny;
        Dim k = index / (Nx * Ny);
        return second_derivative(axes, derivation_direction, i, j, k);
    }

    Real divergence(Dim index) const
    {
        Dim i = index % Nx;
        Dim j = (index / Nx) % Ny;
        Dim k = index / (Nx * Ny);
        return divergence(i, j, k);
    }

    Real divergence(Dim i, Dim j, Dim k) const
    {
        return (first_derivative(0, 0, i, j, k) + first_derivative(1, 1, i, j, k) + first_derivative(2, 2, i, j, k));
    }

    VectorVariable &operator+=(const VectorVariable &rhs)
    {
        if (Nx != rhs.get_Nx() || Ny != rhs.get_Ny() || Nz != rhs.get_Nz())
            throw std::invalid_argument("VectorVariable::operator+= dimension mismatch");

        for (int a = 0; a < static_cast<int>(size()); ++a)
        {
            for (Dim i = 0; i < Nx; ++i)
            {
                for (Dim j = 0; j < Ny; ++j)
                {
                    for (Dim k = 0; k < Nz; ++k)
                    {
                        this->set(a, i, j, k) += rhs.value(a, i, j, k);
                    }
                }
            }
        }
        return *this;
    }

    VectorVariable operator-(const VectorVariable &rhs) const
    {
        if (Nx != rhs.get_Nx() || Ny != rhs.get_Ny() || Nz != rhs.get_Nz())
            throw std::invalid_argument("VectorVariable::operator- dimension mismatch");

        VectorVariable out(Nx, Ny, Nz, dx, dy, dz);
        for (int a = 0; a < static_cast<int>(size()); ++a)
        {
            for (Dim i = 0; i < Nx; ++i)
            {
                for (Dim j = 0; j < Ny; ++j)
                {
                    for (Dim k = 0; k < Nz; ++k)
                    {
                        out.set(a, i, j, k) = value(a, i, j, k) - rhs.value(a, i, j, k);
                    }
                }
            }
        }
        return out;
    }

    VectorVariable operator+(const VectorVariable &rhs) const
    {
        if (Nx != rhs.get_Nx() || Ny != rhs.get_Ny() || Nz != rhs.get_Nz())
            throw std::invalid_argument("VectorVariable::operator+ dimension mismatch");

        VectorVariable out(Nx, Ny, Nz, dx, dy, dz);
        for (int a = 0; a < static_cast<int>(size()); ++a)
        {
            for (Dim i = 0; i < Nx; ++i)
            {
                for (Dim j = 0; j < Ny; ++j)
                {
                    for (Dim k = 0; k < Nz; ++k)
                    {
                        out.set(a, i, j, k) = value(a, i, j, k) + rhs.value(a, i, j, k);
                    }
                }
            }
        }
        return out;
    }

    inline ScalarVariable &component(int axes)
    {
        if (axes < 0 || axes >= static_cast<int>(size()))
            throw std::out_of_range("axes index out of range");
        return data[axes];
    }

    // Convenient named accessors
    inline ScalarVariable &x() { return component(0); }
    inline ScalarVariable &y() { return component(1); }
    inline ScalarVariable &z() { return component(2); }

private:
    std::vector<ScalarVariable> data;
    Dim Nx, Ny, Nz;
    Real dx, dy, dz;
};

#endif // VECTORVARIABLES_HPP