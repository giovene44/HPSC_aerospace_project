#include "Matrix3D.h"
#include <iostream>

int main() {
    std::cout << "=== Matrix3D Test Program ===\n\n";
    
    // Create a 5x6x7 matrix of doubles
    std::cout << "Creating a 5x6x7 matrix of doubles...\n";
    Matrix3D<double> mat(5, 6, 7);
    
    // Fill matrix with some values (for easy identification)
    std::cout << "Filling matrix with test values...\n";
    for (size_t i = 0; i < mat.dim_x(); ++i) {
        for (size_t j = 0; j < mat.dim_y(); ++j) {
            for (size_t k = 0; k < mat.dim_z(); ++k) {
                // Value = i*100 + j*10 + k (easy to read coordinates)
                mat(i, j, k) = i * 100.0 + j * 10.0 + k;
            }
        }
    }
    std::cout << "Matrix dimensions: " << mat.dim_x() << " x " 
              << mat.dim_y() << " x " << mat.dim_z() << "\n\n";
    
    // Test 1: Print default view (first 10x10 of X layers)
    std::cout << "=== Test 1: Default print (X layers, first 5x5) ===\n";
    mat.print(5, 5, 2);  // Show 5x5 slice of first 2 X layers
    
    // Test 2: Print with offsets
    std::cout << "=== Test 2: Print with offsets (X layers 2-3, y=2-4, z=3-5) ===\n";
    mat.print(3, 3, 2, 2, 3, 2);  // len=3, height=3, depth=2, offsets: y=2, z=3, x=2
    
    // Test 3: Print Y layers
    std::cout << "=== Test 3: Y layer view (first 2 Y layers, 4x4 slice) ===\n";
    mat.print(4, 4, 2, 0, 0, 0, 'y');
    
    // Test 4: Print Z layers
    std::cout << "=== Test 4: Z layer view (first 2 Z layers, 4x4 slice) ===\n";
    mat.print(4, 4, 2, 0, 0, 0, 'z');
    
    // Test 5: Access and modify specific elements
    std::cout << "=== Test 5: Element access and modification ===\n";
    std::cout << "Original value at (2, 3, 4): " << mat(2, 3, 4) << "\n";
    mat(2, 3, 4) = 999.99;
    std::cout << "Modified value at (2, 3, 4): " << mat(2, 3, 4) << "\n";
    std::cout << "Printing layer x=2 to see the change:\n";
    mat.print(6, 7, 1, 0, 0, 2);
    
    // Test 6: Create integer matrix
    std::cout << "=== Test 6: Integer matrix (3x4x3) ===\n";
    Matrix3D<int> intMat(3, 4, 3);
    for (size_t i = 0; i < intMat.dim_x(); ++i) {
        for (size_t j = 0; j < intMat.dim_y(); ++j) {
            for (size_t k = 0; k < intMat.dim_z(); ++k) {
                intMat(i, j, k) = i + j + k;
            }
        }
    }
    intMat.print(4, 3, 3);
    
    std::cout << "=== All tests completed successfully! ===\n";
    
    return 0;
}