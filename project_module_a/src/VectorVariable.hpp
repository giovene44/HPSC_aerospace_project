 #include <vector>
 
 class VectorVariable{
    public:
        VectorVariable(int Nx_, int Ny_, int Nz_){
            Nx = Nx_;
            Ny = Ny_;
            Nz = Nz_;
        }

        //accessing method:
        //assuming i,j,k start from 0.
        void value(int axes, int i, int j, int k, float &val){
            if(axes < 0 || axes >= 3) throw std::out_of_range("axes index out of range");
            if(i < 0 || i >= Nx) val = 0.0;
            else if(j < 0 || j >= Ny) val = 0.0;
            else if(k < 0 || k >= Nz) val = 0.0;
            else
            val = data[axes][i + j*Nx + k*Nx*Ny];
        }

        void value(int axes, int index, float &val){
            if(axes < 0 || axes >= 3) throw std::out_of_range("axes index out of range");
            if(index < 0 || index >= Nx*Ny*Nz) throw std::out_of_range("index out of range");
            val = data[axes][index];
        }

        void set_value(int axes, int i, int j, int k, float val){
            data[axes][i + j*Nx + k*Nx*Ny] = val;
        }

        void set_value(int axes, int index, float val){
            data[axes][index] = val;
        }

        void second_derivative(int axes, int derivation_direction, int i, int j, int k, float &val){
           float v1, v2, v3;
           if(derivation_direction == 0){ //x direction
                value(axes, i+1, j, k, v1);
                value(axes, i, j, k, v2);
                value(axes, i-1, j, k, v3);
              }
              else if(derivation_direction == 1){ //y direction
                 value(axes, i, j+1, k, v1);
                 value(axes, i, j, k, v2);
                 value(axes, i, j-1, k, v3);
                  }
                else if(derivation_direction == 2){ //z direction
                   value(axes, i, j, k+1, v1);
                   value(axes, i, j, k, v2);
                   value(axes, i, j, k-1, v3);
                    }
            val = v1 - 2*v2 + v3;

        }

        void second_derivative(int axes, int derivation_direction, int index, float &val){
            int i = index % Nx;
            int j = (index / Nx) % Ny;
            int k = index / (Nx * Ny);
            second_derivative(axes, derivation_direction, i, j, k, val);
        }

        void divergence(int index, float &val){
            int i = index % Nx;
            int j = (index / Nx) % Ny;
            int k = index / (Nx * Ny);
            divergence(i, j, k, val);
        }

        void divergence(int i, int j, int k, float &val){
            float dVx_dx, dVy_dy, dVz_dz;
            second_derivative(0, 0, i, j, k, dVx_dx);
            second_derivative(1, 1, i, j, k, dVy_dy);
            second_derivative(2, 2, i, j, k, dVz_dz);
            val = dVx_dx + dVy_dy + dVz_dz;
        }



    private:

    std::vector<std::vector<float>> data;
    int Nx, Ny, Nz;

        
    };