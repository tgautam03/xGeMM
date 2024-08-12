#include <iostream>

#include <cublas_v2.h>

#include "../include/cublas_sgemm.cuh"

// CUDA Error Checking
#define cuda_check(err) { \
    if (err != cudaSuccess) { \
        std::cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << "\n"; \
        exit(EXIT_FAILURE); \
    } \
}

// CUBLAS Error Checking
#define cublas_check(status) { \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error" << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

void cublas_sgemm(float* A, float* B, float* C, int Nrows_A, int Nrows_B, int Ncols_B)
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

    // Create and initialize cuBLAS handle
    cublasHandle_t handle;
    cublas_check(cublasCreate(&handle));

    // Perform matrix multiplication: C = alpha * A * B + beta * C
    float alpha = 1;
    float beta = 0;
    cublas_check(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             Ncols_B, Nrows_A, Nrows_B, // Num Cols of C, Num rows of C, Shared dim of A and B
                             &alpha,
                             d_B, Ncols_B, // Num cols of B
                             d_A, Nrows_B, // Num cols of A
                             &beta,
                             d_C, Ncols_B) // Num cols of C
                );

    // Copy back results
    cudaError_t err_C_ = cudaMemcpy(C, d_C, Nrows_A*Ncols_B*sizeof(float), cudaMemcpyDeviceToHost);
    cuda_check(err_C_);

    // Free memory
    cublas_check(cublasDestroy(handle));
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
