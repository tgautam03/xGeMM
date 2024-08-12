#ifndef CUBLAS_SGEMM
#define CUBLAS_SGEMM

void cublas_sgemm(float* A, float* B, float* C, int Nrows_A, int Nrows_B, int Ncols_B);

#endif