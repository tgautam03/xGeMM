#ifndef MATRIXFP32
#define MATRIXFP32

class MatrixFP32
{
private:
    int _n_rows;        // Number of rows
    int _n_cols;        // Number of cols

public:
    // Pointer to dynamic array
    float* _mat;

    // Matrix in device memory: true
    bool on_device_;    

    // Constructor to initialize n_rows x n_cols matrix
    MatrixFP32(int n_rows, int n_cols, bool on_device);
    
    // Free memory
    void free_mat();

    // Member function to get matrix dimensions (n_rows, n_cols)
    int rows();
    int cols();

    // Member Function to get value at (i,j)
    float get_val(int row, int col);

    // Member Function to set value (val) at (i,j)
    void set_val(int row, int col, float val);
};

#endif