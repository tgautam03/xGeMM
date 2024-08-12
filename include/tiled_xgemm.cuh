#ifndef TILED_MAT_MUL_GPU
# define TILED_MAT_MUL_GPU

void tiled_mat_mul_kernel(float* A, float* B, float* C, int Nrows_A, int Nrows_B, int Ncols_B);

void tiled_xgemm(float* A, float* B, float* C, int Nrows_A, int Nrows_B, int Ncols_B);

#endif