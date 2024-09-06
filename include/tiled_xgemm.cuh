#ifndef TILED_XGEMM
#define TILED_XGEMM

#include "MatrixFP32.cuh"

void tiled_mat_mul_kernel(MatrixFP32 d_A, MatrixFP32 d_B, MatrixFP32 d_C);

void tiled_xgemm(MatrixFP32 d_A, MatrixFP32 d_B, MatrixFP32 d_C);

#endif