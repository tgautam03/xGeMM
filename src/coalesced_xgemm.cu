#include "../include/MatrixFP32.cuh"
#include <assert.h>

__global__ void coalesced_mat_mul_kernel(MatrixFP32 d_A, MatrixFP32 d_B, MatrixFP32 d_C)
{
    // Working on C[row,col]
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    int row = blockDim.y*blockIdx.y + threadIdx.y;

    // Parallel mat mul
    if (row < d_C.rows() && col < d_C.cols())
    {
        // Value at C[row,col]
        float value = 0;
        for (int k = 0; k < d_B.rows(); k++)
        {
            value += d_A.get_val(row, k) * d_B.get_val(k, col);
        }

        // Assigning calculated value (SGEMM is C = α*(A @ B)+β*C and in this repo α=1, β=0)
        d_C.set_val(row, col, 1*value + 0*d_C.get_val(row, col));
    }
}

void coalesced_xgemm(MatrixFP32 d_A, MatrixFP32 d_B, MatrixFP32 d_C, const int dim_block_x, const int dim_block_y)
{
    // Make sure that matirces are on Device
    assert(d_A._on_device == true && "Matrix must be on device");
    assert(d_B._on_device == true && "Matrix must be on device");
    assert(d_C._on_device == true && "Matrix must be on device");

    // Kernel execution
    dim3 dim_block(dim_block_x, dim_block_y, 1);
    dim3 dim_grid(ceil(d_C.cols()/(float)(dim_block_x)), ceil(d_C.rows()/(float)(dim_block_y)), 1);
    coalesced_mat_mul_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C);
}