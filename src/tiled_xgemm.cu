#include "../include/MatrixFP32.cuh"
#include <assert.h>

#define TILE_WIDTH 32

__global__ void tiled_mat_mul_kernel(MatrixFP32 d_A, MatrixFP32 d_B, MatrixFP32 d_C)
{
    // Ensure that TILE_WIDTH = BLOCK_SIZE
    assert(TILE_WIDTH == blockDim.x);
    assert(TILE_WIDTH == blockDim.y);
    
    // Details regarding this thread
    int by = blockIdx.y;
    int bx = blockIdx.x; 

    int ty = threadIdx.y;
    int tx = threadIdx.x; 

    // Working on C[row,col]
    int row = TILE_WIDTH*by + ty;
    int col = TILE_WIDTH*bx + tx;

    // Allocating shared memory
    __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

    // Parallel mat mul
    float value = 0;
    for (int phase = 0; phase < ceil((float)d_A.cols()/TILE_WIDTH); phase++)
    {
        // Load Tiles into shared memory
        if ((row < d_A.rows()) && ((phase*TILE_WIDTH+tx) < d_A.cols()))
          sh_A[ty][tx] = d_A.get_val(row, phase*TILE_WIDTH+tx);
        else
          sh_A[ty][tx] = 0.0f;

        if (((phase*TILE_WIDTH + ty) < d_B.rows()) && (col < d_B.cols()))
          sh_B[ty][tx] = d_B.get_val(phase*TILE_WIDTH + ty, col);
        else
          sh_B[ty][tx] = 0.0f;
        __syncthreads();

        // Dot product
        for (int k = 0; k < TILE_WIDTH; k++)
            value += sh_A[ty][k] * sh_B[k][tx];
        __syncthreads();
    }
    // Assigning calculated value
    if ((row < d_C.rows()) && (col < d_C.cols()))
        d_C.set_val(row, col, 1*value + 0*d_C.get_val(row, col));
}

void tiled_xgemm(MatrixFP32 d_A, MatrixFP32 d_B, MatrixFP32 d_C, const int dim_block_x, const int dim_block_y)
{
    // Kernel execution
    dim3 dim_block(dim_block_x, dim_block_y, 1);
    dim3 dim_grid(ceil(d_C.cols()/(float)(dim_block_x)), ceil(d_C.rows()/(float)(dim_block_y)), 1);
    tiled_mat_mul_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C);
}