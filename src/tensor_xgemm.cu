#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <cuda_fp16.h>

#include "../include/utils.cuh"

// WMMA fragment dimensions
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void tensor_mat_mul_kernel(half *d_A_ptr, half *d_B_ptr, float *d_C_ptr, int C_n_rows, int C_n_cols, int A_n_cols) 
{
    // Tile using a 2D grid
    int warpM = blockIdx.x; //(blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = blockIdx.y;
    
    // Declare the fragments
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize the output to zero
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    // Loop over A_n_cols
    for (int i = 0; i < A_n_cols; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * WMMA_N;

        // Load the inputs
        nvcuda::wmma::load_matrix_sync(a_frag, &d_A_ptr[aRow * A_n_cols + aCol], A_n_cols);
        nvcuda::wmma::load_matrix_sync(b_frag, &d_B_ptr[bRow * C_n_cols + bCol], C_n_cols);

        // Perform the matrix multiplication
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store the output
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    nvcuda::wmma::store_matrix_sync(&d_C_ptr[cRow * C_n_cols + cCol], c_frag, C_n_cols, nvcuda::wmma::mem_row_major);
}

void tensor_xgemm(half *d_A_ptr, half *d_B_ptr, float *d_C_ptr, int C_n_rows, int C_n_cols, int A_n_cols)
{
    // Kernel execution
    dim3 dim_block(32, 1);
    dim3 dim_grid(C_n_rows / WMMA_M, C_n_cols / WMMA_N);
    tensor_mat_mul_kernel<<<dim_grid, dim_block>>>(d_A_ptr, d_B_ptr, d_C_ptr, C_n_rows, C_n_cols, A_n_cols);
}