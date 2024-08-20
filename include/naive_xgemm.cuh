#ifndef NAIVE_XGEMM
# define NAIVE_XGEMM

void naive_mat_mul_kernel(float* d_A, float* d_B, float* d_C, int Nrows_A, int Nrows_B, int Ncols_B);

void naive_xgemm(float* d_A, float* d_B, float* d_C, int Nrows_A, int Nrows_B, int Ncols_B, const int dim_block_x, const int dim_block_y);

#endif