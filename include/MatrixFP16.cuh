#ifndef MATRIXFP16
#define MATRIXFP16

#include <cuda_fp16.h>

class MatrixFP16
{
public:
    const int n_rows;        // Number of rows
    const int n_cols;        // Number of cols

    // Pointer to dynamic array
    half* ptr;
    
    // Matrix in device memory: true; else: false
    const bool on_device; 
    
    // Constructor to initialize n_rows x n_cols matrix
    MatrixFP16(int n_rows, int n_cols, bool on_device);
    
    // Free memory
    void free_mat();

    // Copy to Device
    void copy_to_device(MatrixFP16 d_mat);

    // Copy to host
    void copy_to_host(MatrixFP16 h_mat);
};

#endif