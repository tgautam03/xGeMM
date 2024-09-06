#include "../include/MatrixFP32.cuh"
#include <assert.h>

__global__ void coalesced_mat_mul_kernel(MatrixFP32 d_A, MatrixFP32 d_B, MatrixFP32 d_C)
{
    // Working on C[row,col]
    const int col = blockDim.x*blockIdx.x + threadIdx.x;
    const int row = blockDim.y*blockIdx.y + threadIdx.y;

    // Matrix cols
    const int A_n_cols = d_A.n_cols;
    const int B_n_cols = d_B.n_cols;
    const int C_n_cols = d_C.n_cols;

    // Parallel mat mul
    if (row < d_C.n_rows && col < d_C.n_cols)
    {
        // Value at C[row,col]
        float value = 0;
        for (int k = 0; k < d_B.n_rows; k++)
        {
            value += d_A.ptr[row*A_n_cols + k] * d_B.ptr[k*B_n_cols + col];
        }

        // Assigning calculated value (SGEMM is C = α*(A @ B)+β*C and in this repo α=1, β=0)
        d_C.ptr[row*C_n_cols + col] = 1*value + 0*d_C.ptr[row*C_n_cols + col];
    }
}

void coalesced_xgemm(MatrixFP32 d_A, MatrixFP32 d_B, MatrixFP32 d_C)
{
    // Make sure that matirces are on Device
    assert(d_A.on_device == true && "Matrix must be on device");
    assert(d_B.on_device == true && "Matrix must be on device");
    assert(d_C.on_device == true && "Matrix must be on device");

    // Kernel execution
    dim3 dim_block(32, 32, 1);
    dim3 dim_grid(ceil(d_C.n_cols/(float)(32)), ceil(d_C.n_rows/(float)(32)), 1);
    coalesced_mat_mul_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C);
}