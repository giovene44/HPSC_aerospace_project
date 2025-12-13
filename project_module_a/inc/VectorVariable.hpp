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
            data.emplace_back(ScalarVariable(Nx, Ny, Nz, dx, dy, dz));
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

            data[0].set(idx) = other.value<0>(x + dx / Real(2.0), y, z, t);
            data[1].set(idx) = other.value<1>(x, y + dy / Real(2.0), z, t);
            data[2].set(idx) = other.value<2>(x, y, z + dz / Real(2.0), t);
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

    // TODO: This should be changed: it needs to shift to the pressure nodes!
    Real first_derivative(int axes, int derivation_direction, Dim i, Dim j, Dim k) const
    {

        Real v_plus = 0.0;
        Real v_minus = 0.0;
        Real den = 0.0;

        // we need to handle the x=0, y=0, z=0 borders outside of this method
        // because they require BC values for velocity!

        if (derivation_direction == 0)
        { // x
            if (i == 0 && axes == 0)
                throw std::invalid_argument("VectorVariable::first_derivative: invalid i for centered difference");

            v_plus = value(axes, i, j, k);
            v_minus = value(axes, i - 1, j, k);
            den = dx;
        }
        else if (derivation_direction == 1)
        { // y
            if (j == 0 && axes == 1)
                throw std::invalid_argument("VectorVariable::first_derivative: invalid j for centered difference");

            v_plus = value(axes, i, j, k);
            v_minus = value(axes, i, j - 1, k);
            den = dy;
        }
        else if (derivation_direction == 2)
        { // z
            if (k == 0 && axes == 2)
                throw std::invalid_argument("VectorVariable::first_derivative: invalid k for centered difference");

            v_plus = value(axes, i, j, k);
            v_minus = value(axes, i, j, k - 1);
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

    // second order derivative on the velocity nodes (staggering handled by using
    // the component index for value(...) access). Keep signature identical but
    // make parameter meanings explicit: axes -> component, derivation_direction -> dir.
    //
    // Usage: second_derivative(component, dir, i, j, k)
    //   component: 0 = u, 1 = v, 2 = w
    //   dir:       0 = d^2/dx^2, 1 = d^2/dy^2, 2 = d^2/dz^2
    //
    // Notes:
    // - Interior points: standard 3-point centered scheme.
    // - Near physical domain boundaries (i==0 or i==Nx-1 etc.) we use a one-sided
    //   4-point second-order accurate formula (same as before).
    // - Ensure Nx,Ny,Nz >= 4 when using one-sided formulas (you already documented that).
    Real second_derivative(int component, int dir, Dim i, Dim j, Dim k) const
    {
        // sanity checks (optional - remove in hot loops if you want performance)
        if (component < 0 || component > 2)
            throw std::invalid_argument("VectorVariable::second_derivative: invalid component index");
        if (dir < 0 || dir > 2)
            throw std::invalid_argument("VectorVariable::second_derivative: invalid derivation direction");

        // choose spacing and indices according to direction
        if (dir == 0) // x-direction
        {
            // boundary one-sided (left)
            if (i == 0)
            {
                // (2 u0 - 5 u1 + 4 u2 - u3) / dx^2  — second-order one-sided
                return (Real(2.0) * value(component, 0, j, k) - Real(5.0) * value(component, 1, j, k) + Real(4.0) * value(component, 2, j, k) - value(component, 3, j, k)) / (dx * dx);
            }
            // boundary one-sided (right)
            if (i == Nx - 1)
            {
                // mirrored one-sided at right end
                return (Real(2.0) * value(component, Nx - 1, j, k) - Real(5.0) * value(component, Nx - 2, j, k) + Real(4.0) * value(component, Nx - 3, j, k) - value(component, Nx - 4, j, k)) / (dx * dx);
            }

            // interior centered
            Real u_ip1 = value(component, i + 1, j, k);
            Real u_i = value(component, i, j, k);
            Real u_im1 = value(component, i - 1, j, k);
            return (u_ip1 - Real(2.0) * u_i + u_im1) / (dx * dx);
        }
        else if (dir == 1) // y-direction
        {
            if (j == 0)
            {
                return (Real(2.0) * value(component, i, 0, k) - Real(5.0) * value(component, i, 1, k) + Real(4.0) * value(component, i, 2, k) - value(component, i, 3, k)) / (dy * dy);
            }
            if (j == Ny - 1)
            {
                return (Real(2.0) * value(component, i, Ny - 1, k) - Real(5.0) * value(component, i, Ny - 2, k) + Real(4.0) * value(component, i, Ny - 3, k) - value(component, i, Ny - 4, k)) / (dy * dy);
            }

            Real u_jp1 = value(component, i, j + 1, k);
            Real u_j = value(component, i, j, k);
            Real u_jm1 = value(component, i, j - 1, k);
            return (u_jp1 - Real(2.0) * u_j + u_jm1) / (dy * dy);
        }
        else // dir == 2, z-direction
        {
            if (k == 0)
            {
                return (Real(2.0) * value(component, i, j, 0) - Real(5.0) * value(component, i, j, 1) + Real(4.0) * value(component, i, j, 2) - value(component, i, j, 3)) / (dz * dz);
            }
            if (k == Nz - 1)
            {
                return (Real(2.0) * value(component, i, j, Nz - 1) - Real(5.0) * value(component, i, j, Nz - 2) + Real(4.0) * value(component, i, j, Nz - 3) - value(component, i, j, Nz - 4)) / (dz * dz);
            }

            Real u_kp1 = value(component, i, j, k + 1);
            Real u_k = value(component, i, j, k);
            Real u_km1 = value(component, i, j, k - 1);
            return (u_kp1 - Real(2.0) * u_k + u_km1) / (dz * dz);
        }
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
        for (Dim a = 0; a < Dim(size()); ++a)
        {
            for (Dim i = 0; i < Nx; ++i)
            {
                for (Dim j = 0; j < Ny; ++j)
                {
                    for (Dim k = 0; k < Nz; ++k)
                    {
                        out.set(a, i, j, k) = this->value(a, i, j, k) - rhs.value(a, i, j, k);
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

    void print_vector()
    {
        for (int a = 0; a < 3; a++)
        {
            std::cout << "Component " << a << ":\n";
            for (Dim k = 0; k < Nz; k++)
            {
                for (Dim j = Ny - 1; j >= 0; j--)
                {
                    for (Dim i = 0; i < Nx; i++)
                    {
                        std::cout << value(a, i, j, k) << " ";
                    }
                    std::cout << "\n";
                }
                std::cout << "----\n";
            }
            std::cout << "====================\n";
        }
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