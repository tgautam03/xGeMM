#include <iostream>
#include <cublas_v2.h>

#include "../include/MatrixFP32.cuh"
#include "../include/utils.hpp"

#include "../include/coarse_1d_xgemm.cuh"

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

int main(int argc, char const *argv[])
{
    int n = 8;

    // Define MatrixFP32
    MatrixFP32 A_FP32 = MatrixFP32(n, n, false);
    MatrixFP32 B_FP32 = MatrixFP32(n, n, false);
    MatrixFP32 C_FP32_cublas = MatrixFP32(n, n, false);
    MatrixFP32 C_FP32_xgemm = MatrixFP32(n, n, false);

    // Initialize Matrices
    init_mat(A_FP32, -10, 10);          // Random Initialization between -10 and 10
    init_mat(B_FP32, -10, 10);          // Random Initialization between -10 and 10
    init_mat(C_FP32_cublas, 1.0f);     // Initialize to 1
    init_mat(C_FP32_xgemm, -1.0f);     // Initialize to -1

    // Move matrices to device
    MatrixFP32 d_A_FP32 = MatrixFP32(n, n, true); 
    A_FP32.copy_to_device(d_A_FP32);
    MatrixFP32 d_B_FP32 = MatrixFP32(n, n, true); 
    B_FP32.copy_to_device(d_B_FP32);
    MatrixFP32 d_C_FP32_cublas = MatrixFP32(n, n, true); 
    C_FP32_cublas.copy_to_device(d_C_FP32_cublas);
    MatrixFP32 d_C_FP32_xgemm = MatrixFP32(n, n, true); 
    C_FP32_xgemm.copy_to_device(d_C_FP32_xgemm);
    cudaDeviceSynchronize();

    //----------------------------------------------------//
    //-------------------- Warmup Run --------------------//
    //----------------------------------------------------//
    // Create and initialize cuBLAS handle
    cublasHandle_t handle;
    cublas_check(cublasCreate(&handle));
    
    // Perform matrix multiplication: C = A * B 
    float alpha = 1;
    float beta = 0;
    cublas_check(cublasSgemm(handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            n, n, n, // Num Cols of C, Num rows of C, Shared dim of A and B
                            &alpha,
                            d_B_FP32._mat, n, // Num cols of B
                            d_A_FP32._mat, n, // Num cols of A
                            &beta,
                            d_C_FP32_cublas._mat, n) // Num cols of C
                );
    cudaDeviceSynchronize();

    // coarse_1d Kernel execution
    coarse_1d_xgemm(d_A_FP32, d_B_FP32, d_C_FP32_xgemm);
    cudaDeviceSynchronize();

    // Printing the smallest matrix result
    d_C_FP32_cublas.copy_to_host(C_FP32_cublas);
    d_C_FP32_xgemm.copy_to_host(C_FP32_xgemm);
    if (n <= 8)
    {
        std::cout << "Matrix C (cuBLAS): \n";
        print_mat(C_FP32_cublas, true);
        std::cout << "\n";

        std::cout << "Matrix C (xGeMM): \n";
        print_mat(C_FP32_xgemm, true);
        std::cout << "\n";
    }

    // Assert that coarse_1d implementation is correct
    std::cout << "Asserting Results for N: " << n << "\n";
    assert_mat(C_FP32_xgemm, C_FP32_cublas, 1e-8);
    std::cout << "Assertion Passed! \n \n";
        
    // Free Memory
    A_FP32.free_mat();
    B_FP32.free_mat();
    C_FP32_cublas.free_mat();
    C_FP32_xgemm.free_mat();

    d_A_FP32.free_mat();
    d_B_FP32.free_mat();
    d_C_FP32_cublas.free_mat();
    d_C_FP32_xgemm.free_mat();

    return 0;
}
