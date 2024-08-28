#ifndef COALESCED_XGEMM
#define COALESCED_XGEMM

#include "MatrixFP32.cuh"

void coalesced_mat_mul_kernel(MatrixFP32 d_A, MatrixFP32 d_B, MatrixFP32 d_C);

void coalesced_xgemm(MatrixFP32 d_A, MatrixFP32 d_B, MatrixFP32 d_C, const int dim_block_x, const int dim_block_y);

#endif