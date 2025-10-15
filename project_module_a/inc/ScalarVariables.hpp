#ifndef SCALARVARIABLES_HPP
#define SCALARVARIABLES_HPP

#include "variables.hpp"
#include <vector>


class ScalarVariables {
    public:
        ScalarVariables(Dim Nx, Dim Ny, Dim Nz)
            : Nx(Nx), Ny(Ny), Nz(Nz) {
                pressure_data.resize(Nx * Ny * Nz, 0.0f);
            };

        Real& set(Dim i, Dim j, Dim k) { //This set works like set(i,j,k) = value;
            return pressure_data[i + j * Nx + k * Nx * Ny];
        }

        Real get(Dim i, Dim j, Dim k) const {
            return pressure_data[i + j * Nx + k * Nx * Ny];
        }
        Real getGradient_x(Dim i, Dim j, Dim k) const {
            return (get(i+1,j,k) - get(i-1,j,k)) / 2.0f;
        }

        Real getGradient_y(Dim i, Dim j, Dim k) const {
            return (get(i,j+1,k) - get(i,j-1,k)) / 2.0f;
        }

        Real getGradient_z(Dim i, Dim j, Dim k) const {
            return (get(i,j,k+1) - get(i,j,k-1)) / 2.0f;
        }

    private:
        const Dim Nx;
        const Dim Ny;
        const Dim Nz;
        std::vector<Real> pressure_data;
};

#endif // SCALARVARIABLES_HPP