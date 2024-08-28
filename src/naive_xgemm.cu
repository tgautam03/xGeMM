#include "../include/MatrixFP32.cuh"
#include <assert.h>

__global__ void naive_mat_mul_kernel(MatrixFP32 d_A, MatrixFP32 d_B, MatrixFP32 d_C, int A_n_rows, int B_n_rows, int B_n_cols)
{
    // Working on C[row,col]
    int row = blockDim.x*blockIdx.x + threadIdx.x;
    int col = blockDim.y*blockIdx.y + threadIdx.y;

    // Parallel mat mul
    if (row < A_n_rows && col < B_n_cols)
    {
        // Value at C[row,col]
        float value = 0;
        for (int k = 0; k < B_n_rows; k++)
        {
            value += d_A.get_val(row, k) * d_B.get_val(k, col);
        }

        // Assigning calculated value
        d_C.set_val(row, col, value);
    }
}

void naive_xgemm(MatrixFP32 d_A, MatrixFP32 d_B, MatrixFP32 d_C, int A_n_rows, int B_n_rows, int B_n_cols, const int dim_block_x, const int dim_block_y)
{
    // Make sure that matirces are on Device
    assert(d_A._on_device == true && "Matrix must be on device");
    assert(d_B._on_device == true && "Matrix must be on device");
    assert(d_C._on_device == true && "Matrix must be on device");

    // Kernel execution
    dim3 dim_block(dim_block_x, dim_block_y, 1);
    dim3 dim_grid(ceil(B_n_cols/(float)(dim_block_x)), ceil(A_n_rows/(float)(dim_block_y)), 1);
    naive_mat_mul_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C, A_n_rows, B_n_rows, B_n_cols);
}