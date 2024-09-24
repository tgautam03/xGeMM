#include "../include/MatrixFP32.cuh"
#include <assert.h>

#define TILE_WIDTH 32

__global__ void tiled_mat_mul_kernel(MatrixFP32 d_A, MatrixFP32 d_B, MatrixFP32 d_C)
{
    // Ensure that TILE_WIDTH = BLOCK_SIZE
    assert(TILE_WIDTH == blockDim.x);
    assert(TILE_WIDTH == blockDim.y);
    
    // Details regarding this thread
    const int by = blockIdx.y;
    const int bx = blockIdx.x; 

    const int ty = threadIdx.y;
    const int tx = threadIdx.x; 

    // Working on C[row,col]
    const int row = TILE_WIDTH*by + ty;
    const int col = TILE_WIDTH*bx + tx;

    // Allocating shared memory
    __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

    // Phases
    const int phases = ceil((float)d_A.n_cols/TILE_WIDTH);

    // Parallel mat mul
    float value = 0;
    for (int phase = 0; phase < phases; phase++)
    {
        // Load Tiles into shared memory
        if ((row < d_A.n_rows) && ((phase*TILE_WIDTH+tx) < d_A.n_cols))
          sh_A[ty][tx] = d_A.ptr[(row)*d_A.n_cols + (phase*TILE_WIDTH+tx)];
        else
          sh_A[ty][tx] = 0.0f;

        if (((phase*TILE_WIDTH + ty) < d_B.n_rows) && (col < d_B.n_cols))
          sh_B[ty][tx] = d_B.ptr[(phase*TILE_WIDTH + ty)*d_B.n_cols + (col)];
        else
          sh_B[ty][tx] = 0.0f;
        __syncthreads();

        // Dot product
        for (int k_phase = 0; k_phase < TILE_WIDTH; k_phase++)
            value += sh_A[ty][k_phase] * sh_B[k_phase][tx];
        __syncthreads();
    }
    // Assigning calculated value
    if ((row < d_C.n_rows) && (col < d_C.n_cols))
        d_C.ptr[(row)*d_C.n_cols + (col)] =  1*value + 0*d_C.ptr[(row)*d_C.n_cols + (col)];
}

void tiled_xgemm(MatrixFP32 d_A, MatrixFP32 d_B, MatrixFP32 d_C)
{
    // Kernel execution
    dim3 dim_block(32, 32, 1);
    dim3 dim_grid(ceil(d_C.n_cols/(float)(32)), ceil(d_C.n_rows/(float)(32)), 1);
    tiled_mat_mul_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C);
}