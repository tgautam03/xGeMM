#ifndef COARSE_2D_TRANSPOSE_XGEMM
#define COARSE_2D_TRANSPOSE_XGEMM

#include "MatrixFP32.cuh"

void coarse_2d_transpose_mat_mul_kernel(float *d_A_ptr, float *d_B_ptr, float *d_C_ptr, int C_n_rows, int C_n_cols, int A_n_cols);

void coarse_2d_transpose_xgemm(float *d_A_ptr, float *d_B_ptr, float *d_C_ptr, int C_n_rows, int C_n_cols, int A_n_cols);

#endif