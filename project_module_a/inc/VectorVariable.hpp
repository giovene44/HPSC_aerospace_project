#ifndef VECTORVARIABLES_HPP
#define VECTORVARIABLES_HPP

#include "variables.hpp"
#include <vector>
#include <iostream>

 
 class VectorVariable{
    public:
        VectorVariable(Dim Nx_, Dim Ny_, Dim Nz_, Real dx_, Real dy_, Real dz_)
        : Nx(Nx_), Ny(Ny_), Nz(Nz_), dx(dx_), dy(dy_), dz(dz_)
        {
            data.resize(3);
            for (int a = 0; a < 3; ++a)
                data[a].resize(Nx * Ny * Nz, 0.0);
        }
  

        //accessing method:
        //assuming i,j,k start from 0.
        inline Real value(int axes, Dim i, Dim j, Dim k) const {
            if(axes < 0 || axes >= 3) throw std::out_of_range("axes index out of range");
            if(i < 0 || i >= Nx) return 0.0;
            else if(j < 0 || j >= Ny) return 0.0;
            else if(k < 0 || k >= Nz) return 0.0;
            else
            return data[axes][i + j*Nx + k*Nx*Ny];
        }

        inline Real value(int axes, Dim index) const {
            if(axes < 0 || axes >= 3) throw std::out_of_range("axes index out of range");
            if(index < 0 || index >= Nx*Ny*Nz) throw std::out_of_range("index out of range");
            return data[axes][index];
        }

        inline Real& set(int axes, Dim i, Dim j, Dim k) noexcept {
            return data[axes][i + j * Nx + k * Nx * Ny];
        }

        inline Real& set(int axes, Dim index) noexcept {
            return data[axes][index];
        }


        Real first_derivative(int axes, int derivation_direction, Dim i, Dim j, Dim k) const {
            Real v1, v2, den, val;
            if(derivation_direction == 0){ //x direction
                v1 = value(axes, i+1, j, k);  
                v2 = value(axes, i-1, j, k);
                den = 2*dx;  //uses a centered finite differences scheme
              }
              else if(derivation_direction == 1){ //y direction
                 v1 = value(axes, i, j+1, k);
                 v2 = value(axes, i, j-1, k);
                    den = 2*dy;
                  }
                else if(derivation_direction == 2){ //z direction
                   v1 = value(axes, i, j, k+1);
                   v2 = value(axes, i, j, k-1);
                     den = 2*dz;
                    }
            val = v1 - v2;
            val/=den;
            return val;
        }

        Real first_derivative(int axes, int derivation_direction, Dim index) const {
            Dim i = index % Nx;
            Dim j = (index / Nx) % Ny;
            Dim k = index / (Nx * Ny);
            return first_derivative(axes, derivation_direction, i, j, k);
        }

        Real second_derivative(int axes, int derivation_direction, Dim i, Dim j, Dim k) const {
           Real v1, v2, v3, den, val;
           if(derivation_direction == 0){ //x direction
                v1 = value(axes, i+1, j, k);
                v2 = value(axes, i, j, k);
                v3 = value(axes, i-1, j, k);
                den = dx*dx;
              }
              else if(derivation_direction == 1){ //y direction
                 v1 = value(axes, i, j+1, k);
                 v2 = value(axes, i, j, k);
                 v3 = value(axes, i, j-1, k);
                    den = dy*dy;
                  }
                else if(derivation_direction == 2){ //z direction
                   v1 = value(axes, i, j, k+1);
                   v2 = value(axes, i, j, k);
                   v3 = value(axes, i, j, k-1);
                     den = dz*dz;
                    }
            val = v1 - 2*v2 + v3;
            val/=den;
            return val;
        }

        Real second_derivative(int axes, int derivation_direction, Dim index) const {
            Dim i = index % Nx;
            Dim j = (index / Nx) % Ny;
            Dim k = index / (Nx * Ny);
            return second_derivative(axes, derivation_direction, i, j, k);
        }

        Real divergence(Dim index) const {
            Dim i = index % Nx;
            Dim j = (index / Nx) % Ny;
            Dim k = index / (Nx * Ny);
            return divergence(i, j, k);
        }

        Real divergence(Dim i, Dim j, Dim k) const {
            return (first_derivative(0, 0, i, j, k)
            + first_derivative(1, 1, i, j, k)
            +first_derivative(2, 2, i, j, k));
        }


    private:

    std::vector<std::vector<Real>> data;
    Dim Nx, Ny, Nz;
    Real dx,dy,dz;

        
    };



#endif // VECTORVARIABLES_HPP