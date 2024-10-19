#ifndef MATRIXFP32
#define MATRIXFP32

class MatrixFP32
{
public:
    const int n_rows;        // Number of rows
    const int n_cols;        // Number of cols

    // Pointer to dynamic array
    float* ptr;
    
    // Matrix in device memory: true; else: false
    const bool on_device; 
    
    // Constructor to initialize n_rows x n_cols matrix
    MatrixFP32(int n_rows, int n_cols, bool on_device);
    
    // Free memory
    void free_mat();

    // Copy to Device
    void copy_to_device(MatrixFP32 d_mat);

    // Copy to host
    void copy_to_host(MatrixFP32 h_mat);
};

#endif