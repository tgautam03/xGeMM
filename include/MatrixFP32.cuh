#ifndef MATRIXFP32
#define MATRIXFP32

#include <device_launch_parameters.h>

class MatrixFP32
{
private:
    int _n_rows;        // Number of rows
    int _n_cols;        // Number of cols

public: 
    // Pointer to dynamic array
    float* _mat;
    
    // Matrix in device memory: true; else: false
    bool _on_device; 
    
    // Constructor to initialize n_rows x n_cols matrix
    MatrixFP32(int n_rows, int n_cols, bool on_device);
    
    // Free memory
    void free_mat();

    // Member function to get matrix dimensions (n_rows, n_cols)
    __host__ __device__ int rows() const;
    __host__ __device__ int cols() const;

    // Member Function to get value at (i,j)
    __host__ __device__ float get_val(int row, int col) const;

    // Member Function to set value (val) at (i,j)
    __host__ __device__ void set_val(int row, int col, float val);

    // Copy to Device
    MatrixFP32 copy_to_device();

    // Copy to host
    void copy_to_host(MatrixFP32 h_mat);
};

#endif