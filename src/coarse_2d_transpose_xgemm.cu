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

__global__ void coarse_2d_transpose_mat_mul_kernel(float *d_A_ptr, float *d_B_ptr, float *d_C_ptr, int C_n_rows, int C_n_cols, int A_n_cols)
{
    // Number of threads per block
    const int n_threads_per_block = tiles_A_rows * tiles_B_cols / (COARSE_FACTOR_X*COARSE_FACTOR_Y);
    static_assert(n_threads_per_block % tiles_A_cols == 0);
    static_assert(n_threads_per_block % tiles_B_cols == 0);

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
    __shared__ float sh_A[tiles_A_cols][tiles_A_rows];
    __shared__ float sh_B[tiles_A_cols][tiles_B_cols];

    // Parallel mat mul
    float value[COARSE_FACTOR_Y][COARSE_FACTOR_X] = {0.0f};
    float register_A[COARSE_FACTOR_X] = {0.0f};
    float register_B[COARSE_FACTOR_Y] = {0.0f};

    // Phases
    const int phases = ceil((float)A_n_cols/tiles_A_cols);

    for (int phase = 0; phase < phases; phase++)
    {
        // Load Tiles into shared memory
        for (int load_offset = 0; load_offset < tiles_A_rows; load_offset+=stride_A)
        {
            if ((by*tiles_A_rows + load_offset+A_view_ty < C_n_rows) && ((phase*tiles_A_cols+A_view_tx) < A_n_cols))
                sh_A[A_view_tx][load_offset+A_view_ty] = d_A_ptr[(by*tiles_A_rows+load_offset+A_view_ty)*A_n_cols + (phase*tiles_A_cols+A_view_tx)];
            else
                sh_A[A_view_tx][load_offset+A_view_ty] = 0.0f;
        }
        
        for (int load_offset = 0; load_offset < tiles_A_cols; load_offset+=stride_B)
        {
            if (((phase*tiles_A_cols + B_view_ty+load_offset) < A_n_cols) && (bx*tiles_B_cols + B_view_tx < C_n_cols))
                sh_B[B_view_ty+load_offset][B_view_tx] = d_B_ptr[(phase*tiles_A_cols+B_view_ty+load_offset)*C_n_cols + (bx*tiles_B_cols+B_view_tx)];
            else
                sh_B[B_view_ty+load_offset][B_view_tx] = 0.0f;
        }
        __syncthreads();

        // calculate per-thread results
        for (int k = 0; k < tiles_A_cols; ++k) 
        {
            // block into registers
            for (int i = 0; i < COARSE_FACTOR_Y; ++i)
                register_A[i] = sh_A[k][row+i];
            
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
            if ((by*tiles_A_rows+row+cy < C_n_rows) && (bx*tiles_B_cols+col+cx < C_n_cols))
                d_C_ptr[(by*tiles_A_rows+row+cy)*C_n_cols + (bx*tiles_B_cols+col+cx)] = 1*value[cy][cx] + 0*d_C_ptr[(by*tiles_A_rows+row+cy)*C_n_cols + (bx*tiles_B_cols+col+cx)];
        }
    } 
}

void coarse_2d_transpose_xgemm(float *d_A_ptr, float *d_B_ptr, float *d_C_ptr, int C_n_rows, int C_n_cols, int A_n_cols)
{
    // Kernel execution
    dim3 dim_grid(ceil(C_n_cols/(float)(tiles_B_cols)), ceil(C_n_rows/(float)(tiles_A_rows)));
    dim3 dim_block(tiles_A_rows*tiles_B_cols/(COARSE_FACTOR_X*COARSE_FACTOR_Y));
    coarse_2d_transpose_mat_mul_kernel<<<dim_grid, dim_block>>>(d_A_ptr, d_B_ptr, d_C_ptr, C_n_rows, C_n_cols, A_n_cols);
}