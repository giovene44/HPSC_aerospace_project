#ifndef MATRIX3D_H_
#define MATRIX3D_H_

#include <vector>
#include <cstddef>
#include <iostream>
#include <iomanip>

// Da rifare
/**
 * Generic 3D Matrix class
 * Memory layout: Row-major order
 * Linear index = i * (dim_y * dim_z) + j * dim_z + k
 */
template <typename T>
class Matrix3D
{
private:
    std::vector<T> data_;
    size_t dim_x_;
    size_t dim_y_;
    size_t dim_z_;

    // Convert 3D indices to 1D index
    inline size_t index(size_t i, size_t j, size_t k) const
    {
        return i * (dim_y_ * dim_z_) + j * dim_z_ + k;
    }

public:
    // Constructor: creates matrix with given dimensions
    Matrix3D(size_t dim_x, size_t dim_y, size_t dim_z);

    // Access element for reading or writing
    T &operator()(size_t i, size_t j, size_t k);

    // Access element for reading only
    const T &operator()(size_t i, size_t j, size_t k) const;

    // Get dimensions
    size_t dim_x() const { return dim_x_; }
    size_t dim_y() const { return dim_y_; }
    size_t dim_z() const { return dim_z_; }

    // Print matrix slice
    void print(size_t len = 10, size_t height = 10, size_t depth = 10,
               size_t len_offset = 0, size_t height_offset = 0, size_t depth_offset = 0,
               char layer_direction = 'x') const;
};

// Constructor: allocates memory for all elements
template <typename T>
Matrix3D<T>::Matrix3D(size_t dim_x, size_t dim_y, size_t dim_z)
    : data_(dim_x * dim_y * dim_z),
      dim_x_(dim_x),
      dim_y_(dim_y),
      dim_z_(dim_z) {}

// Access operator (non-const): allows modification
template <typename T>
T &Matrix3D<T>::operator()(size_t i, size_t j, size_t k)
{
    return data_[index(i, j, k)];
}

// Access operator (const): read-only access
template <typename T>
const T &Matrix3D<T>::operator()(size_t i, size_t j, size_t k) const
{
    return data_[index(i, j, k)];
}

// Print function: displays a slice of the matrix
template <typename T>
void Matrix3D<T>::print(size_t len, size_t height, size_t depth,
                        size_t len_offset, size_t height_offset, size_t depth_offset,
                        char layer_direction) const
{
    if (layer_direction == 'x' || layer_direction == 'X')
    {
        // Layers along X axis (original behavior)
        size_t start_x = depth_offset;
        size_t end_x = std::min(depth_offset + depth, dim_x_);
        size_t start_y = len_offset;
        size_t end_y = std::min(len_offset + len, dim_y_);
        size_t start_z = height_offset;
        size_t end_z = std::min(height_offset + height, dim_z_);

        for (size_t i = start_x; i < end_x; ++i)
        {
            std::cout << "Layer x = " << i << ":\n";
            std::cout << "     ";
            for (size_t j = start_y; j < end_y; ++j)
            {
                std::cout << std::setw(10) << "y=" + std::to_string(j);
            }
            std::cout << "\n";

            for (size_t k = start_z; k < end_z; ++k)
            {
                std::cout << "z=" << std::setw(2) << k << " ";
                for (size_t j = start_y; j < end_y; ++j)
                {
                    std::cout << std::setw(10) << (*this)(i, j, k);
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    }
    else if (layer_direction == 'y' || layer_direction == 'Y')
    {
        // Layers along Y axis
        size_t start_y = depth_offset;
        size_t end_y = std::min(depth_offset + depth, dim_y_);
        size_t start_x = len_offset;
        size_t end_x = std::min(len_offset + len, dim_x_);
        size_t start_z = height_offset;
        size_t end_z = std::min(height_offset + height, dim_z_);

        for (size_t j = start_y; j < end_y; ++j)
        {
            std::cout << "Layer y = " << j << ":\n";
            std::cout << "     ";
            for (size_t i = start_x; i < end_x; ++i)
            {
                std::cout << std::setw(10) << "x=" + std::to_string(i);
            }
            std::cout << "\n";

            for (size_t k = start_z; k < end_z; ++k)
            {
                std::cout << "z=" << std::setw(2) << k << " ";
                for (size_t i = start_x; i < end_x; ++i)
                {
                    std::cout << std::setw(10) << (*this)(i, j, k);
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    }
    else if (layer_direction == 'z' || layer_direction == 'Z')
    {
        // Layers along Z axis
        size_t start_z = depth_offset;
        size_t end_z = std::min(depth_offset + depth, dim_z_);
        size_t start_x = len_offset;
        size_t end_x = std::min(len_offset + len, dim_x_);
        size_t start_y = height_offset;
        size_t end_y = std::min(height_offset + height, dim_y_);

        for (size_t k = start_z; k < end_z; ++k)
        {
            std::cout << "Layer z = " << k << ":\n";
            std::cout << "     ";
            for (size_t i = start_x; i < end_x; ++i)
            {
                std::cout << std::setw(10) << "x=" + std::to_string(i);
            }
            std::cout << "\n";

            for (size_t j = start_y; j < end_y; ++j)
            {
                std::cout << "y=" << std::setw(2) << j << " ";
                for (size_t i = start_x; i < end_x; ++i)
                {
                    std::cout << std::setw(10) << (*this)(i, j, k);
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    }
    else
    {
        std::cout << "Invalid layer direction. Use 'x', 'y', or 'z'.\n";
    }
}

#endif // MATRIX3D_H_
