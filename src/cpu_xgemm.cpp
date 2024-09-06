#include "../include/MatrixFP32.cuh"
#include <assert.h>

void cpu_xgemm(MatrixFP32 A_mat, MatrixFP32 B_mat, MatrixFP32 C_mat)
{
    // Getting A Matrix Dimension
    int A_n_rows = A_mat.n_rows; 
    int A_n_cols = A_mat.n_cols;

    // Getting B Matrix Dimension
    int B_n_rows = B_mat.n_rows; 
    int B_n_cols = B_mat.n_cols;

    // Getting C Matrix Dimension
    int C_n_rows = C_mat.n_rows; 
    int C_n_cols = C_mat.n_cols;

    // Asserting dimensions
    assert (A_n_cols == B_n_rows && "Matrices A & B must have one common dimension");
    assert (A_n_rows == C_n_rows && "A rows must be equal to C rows");
    assert (B_n_cols == C_n_cols && "B cols must be equal to C cols");

    for (int row = 0; row < A_n_rows; row++)
    {
        for (int col = 0; col < B_n_cols; col++)
        {
            float val = 0.0f;
            for (int k = 0; k < A_n_cols; k++)
            {
                val += A_mat.ptr[row*A_n_cols + k] * B_mat.ptr[k*B_n_cols + col];
            }
            C_mat.ptr[row*C_n_cols + col] = val;
        }
    }
}