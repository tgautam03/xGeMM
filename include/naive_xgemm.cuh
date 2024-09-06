#ifndef NAIVE_XGEMM
#define NAIVE_XGEMM

#include "MatrixFP32.cuh"

void naive_mat_mul_kernel(MatrixFP32 d_A, MatrixFP32 d_B, MatrixFP32 d_C);

void naive_xgemm(MatrixFP32 d_A, MatrixFP32 d_B, MatrixFP32 d_C);

#endif