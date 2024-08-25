#ifndef MATRIXFP32
#define MATRIXFP32

class MatrixFP32
{
private:
    float* _mat; // Pointer to dynamic array
    int _n_rows; // Number of rows
    int _n_cols; // Number of cols

public:
    // Constructor to initialize n_rows x n_cols matrix
    MatrixFP32(int n_rows, int n_cols);
    // // Deconstructor to free memory
    // ~MatrixFP32();

    // Member function to get matrix dimensions (n_rows, n_cols)
    void shape(int& n_rows, int& n_cols);

    // Member Function to get value at (i,j)
    float get_val(int row, int col);

    // Member Function to set value (val) at (i,j)
    void set_val(int row, int col, float val);
};

#endif