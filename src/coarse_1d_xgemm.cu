#include "../include/MatrixFP32.cuh"
#include <assert.h>

#include <iostream>

// Coalescing Factor
#define COARSE_FACTOR 8

// Tiles of A
#define tiles_A_rows 64
#define tiles_A_cols 8

// Tiles of B
#define tiles_B_cols 64

__global__ void coarse_1d_mat_mul_kernel(MatrixFP32 d_A, MatrixFP32 d_B, MatrixFP32 d_C)
{
  // Details regarding this thread
  int by = blockIdx.y;
  int bx = blockIdx.x; 

  int tx = threadIdx.x; 

  // 1D -> 2D while loading A
  int A_view_ty = tx / tiles_A_cols;
  int A_view_tx = tx % tiles_A_cols;

  // 1D -> 2D while loading B
  int B_view_ty = tx / tiles_B_cols;
  int B_view_tx = tx % tiles_B_cols;

  // Working on C[row,col]
  int row = tiles_A_rows*by + COARSE_FACTOR * (tx/tiles_B_cols);
  int col = tiles_B_cols*bx + (tx % tiles_B_cols);

  // Allocating shared memory
  __shared__ float sh_A[tiles_A_rows][tiles_A_cols];
  __shared__ float sh_B[tiles_A_cols][tiles_B_cols];

  // Parallel mat mul
  float value[COARSE_FACTOR] = {0.0f};
  for (int phase = 0; phase < ceil((float)d_A.cols()/tiles_A_cols); phase++)
  {
    // Load Tiles into shared memory
    if ((by*tiles_A_rows + A_view_ty < d_A.rows()) && ((phase*tiles_A_cols+A_view_tx) < d_A.cols()))
      sh_A[A_view_ty][A_view_tx] = d_A.get_val(by*tiles_A_rows + A_view_ty, phase*tiles_A_cols+A_view_tx);
    else
      sh_A[A_view_ty][A_view_tx] = 0.0f;
    
    if (((phase*tiles_A_cols + B_view_ty) < d_B.rows()) && (bx*tiles_B_cols + B_view_tx < d_B.cols()))
      sh_B[B_view_ty][B_view_tx] = d_B.get_val(phase*tiles_A_cols + B_view_ty, bx*tiles_B_cols + B_view_tx);
    else
      sh_B[B_view_ty][B_view_tx] = 0.0f;
    __syncthreads();

    for (int k = 0; k < tiles_A_cols; k++)
    {
      float B_val_register = sh_B[k][B_view_tx];
      // Dot product
      for (int c = 0; c < COARSE_FACTOR; c++)
        value[c] += sh_A[B_view_ty*COARSE_FACTOR+c][k] * B_val_register;  
    }
    __syncthreads();
  }

  // Assigning calculated value
  for (int c = 0; c < COARSE_FACTOR; ++c)
  {
    if ((row+c < d_C.rows()) && (col < d_C.cols()))
      d_C.set_val(row+c, col, 1*value[c] + 0*d_C.get_val(row+c, col));
  } 
}

void coarse_1d_xgemm(MatrixFP32 d_A, MatrixFP32 d_B, MatrixFP32 d_C)
{
  // Kernel execution
  dim3 dim_grid(ceil(d_C.cols()/(float)(tiles_B_cols)), ceil(d_C.rows()/(float)(tiles_A_rows)));
  dim3 dim_block(tiles_A_rows*tiles_B_cols/COARSE_FACTOR);
  coarse_1d_mat_mul_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C);
}