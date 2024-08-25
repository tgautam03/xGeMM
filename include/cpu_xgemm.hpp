#ifndef CPU_XGEMM
#define CPU_XGEMM

#include "MatrixFP32.cuh"

void cpu_xgemm(MatrixFP32 A_mat, MatrixFP32 B_mat, MatrixFP32 C_mat);

#endif