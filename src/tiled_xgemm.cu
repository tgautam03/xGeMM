#include <iostream>
#include <stdio.h>
#include <assert.h>

// CUDA Error Checking
#define cuda_check(err) { \
    if (err != cudaSuccess) { \
        std::cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << "\n"; \
        exit(EXIT_FAILURE); \
    } \
}

#define TILE_WIDTH 32

__global__ void tiled_mat_mul_kernel(float* A, float* B, float* C, int Nrows_A, int Nrows_B, int Ncols_B)
{
    // Ensure that TILE_WIDTH = BLOCK_SIZE
    assert(TILE_WIDTH == blockDim.x);
    assert(TILE_WIDTH == blockDim.y);
    
    // Details regarding this thread
    int by = blockIdx.y;
    int bx = blockIdx.x; 

    int ty = threadIdx.y;
    int tx = threadIdx.x; 

    // Working on C[i,j]
    int i = TILE_WIDTH*by + ty;
    int j = TILE_WIDTH*bx + tx;

    // Allocating shared memory
    __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

    // Parallel mat mul
    float value = 0;
    for (int phase = 0; phase < ceil((float)Nrows_B/TILE_WIDTH); phase++)
    {
        // Load Tiles into shared memory
        if ((i < Nrows_A) && ((phase*TILE_WIDTH+tx) < Nrows_B))
          sh_A[ty][tx] = A[(i)*Nrows_B + phase*TILE_WIDTH+tx];
        else
          sh_A[ty][tx] = 0.0f;

        if (((phase*TILE_WIDTH + ty) < Nrows_B) && (j < Ncols_B))
          sh_B[ty][tx] = B[(phase*TILE_WIDTH + ty)*Ncols_B+j];
        else
          sh_B[ty][tx] = 0.0f;
        __syncthreads();

        // Dot product
        for (int k = 0; k < TILE_WIDTH; k++)
            value += sh_A[ty][k] * sh_B[k][tx];
        __syncthreads();
    }
    // Assigning calculated value
    if ((i < Nrows_A) && (j < Ncols_B))
      C[i*Ncols_B+j] = value;
}

void tiled_xgemm(float* A, float* B, float* C, int Nrows_A, int Nrows_B, int Ncols_B)
{
    // Device array pointers
    float* d_A;
    float* d_B;
    float* d_C;

    // Device memory allocation
    cudaError_t err_A = cudaMalloc((void**) &d_A, Nrows_A*Nrows_B*sizeof(float));
    cuda_check(err_A);

    cudaError_t err_B = cudaMalloc((void**) &d_B, Nrows_B*Ncols_B*sizeof(float));
    cuda_check(err_B);

    cudaError_t err_C = cudaMalloc((void**) &d_C, Nrows_A*Ncols_B*sizeof(float));
    cuda_check(err_C);

    // Copying A and B to device memory
    cudaError_t err_A_ = cudaMemcpy(d_A, A, Nrows_A*Nrows_B*sizeof(float), cudaMemcpyHostToDevice);
    cuda_check(err_A_);

    cudaError_t err_B_ = cudaMemcpy(d_B, B, Nrows_B*Ncols_B*sizeof(float), cudaMemcpyHostToDevice);
    cuda_check(err_B_);

    // Kernel execution
    dim3 dim_block(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dim_grid(ceil(Ncols_B/(float)(TILE_WIDTH)), ceil(Nrows_A/(float)(TILE_WIDTH)), 1);
    tiled_mat_mul_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C, Nrows_A, Nrows_B, Ncols_B);

    // Copy back results
    cudaError_t err_C_ = cudaMemcpy(C, d_C, Nrows_A*Ncols_B*sizeof(float), cudaMemcpyDeviceToHost);
    cuda_check(err_C_);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}