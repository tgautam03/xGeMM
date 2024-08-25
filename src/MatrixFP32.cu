#include "../include/MatrixFP32.cuh"

MatrixFP32::MatrixFP32(int n_rows, int n_cols)
{
    // Assigning Number of rows and cols to provate variables
    _n_rows = n_rows;
    _n_cols = n_cols;

    // Initialize dynamic array
    _mat = new float[_n_rows*_n_cols];
}

// MatrixFP32::~MatrixFP32()
// {
//     delete[] _mat;
// }

void MatrixFP32::shape(int& n_rows, int& n_cols)
{
    n_rows = _n_rows;
    n_cols = _n_cols;
}

float MatrixFP32::get_val(int row, int col)
{
    return _mat[row*_n_cols + col];
}

void MatrixFP32::set_val(int row, int col, float val)
{
    _mat[row*_n_cols + col] = val;
}
