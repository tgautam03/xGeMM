#include "../include/MatrixFP32.cuh"
#include <assert.h>

#include <iostream>

// Coalescing Factor
#define COARSE_FACTOR_X 8
#define COARSE_FACTOR_Y 8

// Tiles of A
#define tiles_A_rows 128
#define tiles_A_cols 16

// Tiles of B
#define tiles_B_cols 128

__global__ void coarse_2d_mat_mul_kernel(MatrixFP32 d_A, MatrixFP32 d_B, MatrixFP32 d_C)
{
    // Number of threads per block
    const int n_threads_per_block = tiles_A_rows * tiles_B_cols / (COARSE_FACTOR_X*COARSE_FACTOR_Y);
    static_assert(n_threads_per_block % tiles_A_cols == 0);

    // Details regarding this thread
    const int by = blockIdx.y;
    const int bx = blockIdx.x; 

    const int tx = threadIdx.x;

    // 1D -> 2D while loading A
    const int A_view_ty = tx / tiles_A_cols;
    const int A_view_tx = tx % tiles_A_cols;
    const int stride_A = n_threads_per_block/tiles_A_cols;

    // 1D -> 2D while loading B
    const int B_view_ty = tx / tiles_B_cols;
    const int B_view_tx = tx % tiles_B_cols;
    const int stride_B = n_threads_per_block/tiles_B_cols;

    // Working on C[row, col]
    const int row = COARSE_FACTOR_Y * (tx / (tiles_B_cols/COARSE_FACTOR_X));
    const int col = COARSE_FACTOR_X * (tx % (tiles_B_cols/COARSE_FACTOR_X));
    
    // Allocating shared memory
    __shared__ float sh_A[tiles_A_rows][tiles_A_cols];
    __shared__ float sh_B[tiles_A_cols][tiles_B_cols];

    // Parallel mat mul
    float value[COARSE_FACTOR_Y][COARSE_FACTOR_X] = {0.0f};
    float register_A[COARSE_FACTOR_X] = {0.0f};
    float register_B[COARSE_FACTOR_Y] = {0.0f};

    // Phases
    const int phases = ceil((float)d_A.n_cols/tiles_A_cols);

    for (int phase = 0; phase < phases; phase++)
    {
        // Load Tiles into shared memory
        for (int load_offset = 0; load_offset < tiles_A_rows; load_offset+=stride_A)
        {
            if ((by*tiles_A_rows + load_offset+A_view_ty < d_A.n_rows) && ((phase*tiles_A_cols+A_view_tx) < d_A.n_cols))
                sh_A[load_offset+A_view_ty][A_view_tx] = d_A.ptr[(by*tiles_A_rows+load_offset+A_view_ty)*d_A.n_cols + (phase*tiles_A_cols+A_view_tx)];
            else
                sh_A[load_offset+A_view_ty][A_view_tx] = 0.0f;
        }
        
        for (int load_offset = 0; load_offset < tiles_A_cols; load_offset+=stride_B)
        {
            if (((phase*tiles_A_cols + B_view_ty+load_offset) < d_B.n_rows) && (bx*tiles_B_cols + B_view_tx < d_B.n_cols))
                sh_B[B_view_ty+load_offset][B_view_tx] = d_B.ptr[(phase*tiles_A_cols+B_view_ty+load_offset)*d_A.n_cols + (bx*tiles_B_cols+B_view_tx)];
            else
                sh_B[B_view_ty+load_offset][B_view_tx] = 0.0f;
        }
        __syncthreads();

        // calculate per-thread results
        for (int k = 0; k < tiles_A_cols; ++k) 
        {
            // block into registers
            for (int i = 0; i < COARSE_FACTOR_Y; ++i)
                register_A[i] = sh_A[row+i][k];
            
            for (int i = 0; i < COARSE_FACTOR_X; ++i)
                register_B[i] = sh_B[k][col+i];
            
            for (int cy = 0; cy < COARSE_FACTOR_Y; ++cy) 
            {
                for (int cx = 0; cx < COARSE_FACTOR_X; ++cx) 
                    value[cy][cx] += register_A[cy] * register_B[cx];
            }
        }
        __syncthreads();
    }

    // Assigning calculated value
    for (int cy = 0; cy < COARSE_FACTOR_Y; ++cy)
    {
        for (int cx = 0; cx < COARSE_FACTOR_X; cx++)
        {
            if ((by*tiles_A_rows+row+cy < d_C.n_rows) && (bx*tiles_B_cols+col+cx < d_C.n_cols))
                d_C.ptr[(by*tiles_A_rows+row+cy)*d_C.n_cols + (bx*tiles_B_cols+col+cx)] = 1*value[cy][cx] + 0*d_C.ptr[(by*tiles_A_rows+row+cy)*d_C.n_cols + (bx*tiles_B_cols+col+cx)];
        }
    } 
}

void coarse_2d_xgemm(MatrixFP32 d_A, MatrixFP32 d_B, MatrixFP32 d_C)
{
    // Kernel execution
    dim3 dim_grid(ceil(d_C.n_cols/(float)(tiles_B_cols)), ceil(d_C.n_rows/(float)(tiles_A_rows)));
    dim3 dim_block(tiles_A_rows*tiles_B_cols/(COARSE_FACTOR_X*COARSE_FACTOR_Y));
    coarse_2d_mat_mul_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C);
}