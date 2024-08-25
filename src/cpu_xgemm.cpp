#include "../include/MatrixFP32.cuh"
#include <assert.h>

void cpu_xgemm(MatrixFP32 A_mat, MatrixFP32 B_mat, MatrixFP32 C_mat)
{
    // Getting A Matrix Dimension
    int A_n_rows, A_n_cols;
    A_mat.shape(A_n_rows, A_n_cols);

    // Getting B Matrix Dimension
    int B_n_rows, B_n_cols;
    B_mat.shape(B_n_rows, B_n_cols);

    // Getting C Matrix Dimension
    int C_n_rows, C_n_cols;
    C_mat.shape(C_n_rows, C_n_cols);

    // Asserting dimensions
    assert (A_n_cols == B_n_rows && "Matrices A & B must have one common dimension");
    assert (A_n_rows == C_n_rows && "A rows must be equal to C rows");
    assert (B_n_cols == C_n_cols && "B cols must be equal to C cols");

    for (int i = 0; i < A_n_rows; i++)
    {
        for (int j = 0; j < B_n_cols; j++)
        {
            float val = 0.0f;
            for (int k = 0; k < A_n_cols; k++)
            {
                val += A_mat.get_val(i, k) * B_mat.get_val(k, j);
            }
            C_mat.set_val(i, j, val);
        }
    }
}