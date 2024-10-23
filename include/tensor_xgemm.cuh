#ifndef TENSOR_XGEMM
#define TENSOR_XGEMM

#include "MatrixFP32.cuh"
#include <cuda_fp16.h>

void tensor_mat_mul_kernel(half *d_A_ptr, half *d_B_ptr, float *d_C_ptr, int C_n_rows, int C_n_cols, int A_n_cols);

void tensor_xgemm(half *d_A_ptr, half *d_B_ptr, float *d_C_ptr, int C_n_rows, int C_n_cols, int A_n_cols);

#endif