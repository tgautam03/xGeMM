#ifndef COARSE_1D_XGEMM
#define COARSE_1D_XGEMM

#include "MatrixFP32.cuh"

void coarse_1d_mat_mul_kernel(float *d_A_ptr, float *d_B_ptr, float *d_C_ptr, int C_n_rows, int C_n_cols, int A_n_cols);

void coarse_1d_xgemm(float *d_A_ptr, float *d_B_ptr, float *d_C_ptr, int C_n_rows, int C_n_cols, int A_n_cols);

#endif