#ifndef COARSE_1D_XGEMM
#define COARSE_1D_XGEMM

#include "MatrixFP32.cuh"

void coarse_1d_mat_mul_kernel(MatrixFP32 d_A, MatrixFP32 d_B, MatrixFP32 d_C);

void coarse_1d_xgemm(MatrixFP32 d_A, MatrixFP32 d_B, MatrixFP32 d_C);

#endif