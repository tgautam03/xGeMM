#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <random>

#include <cublas_v2.h>

#include "../include/tiled_xgemm.cuh"
#include "../include/utils.hpp"

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

#define MAX_NUM 10 
#define MIN_NUM -10

int main(int argc, char const *argv[])
{
    int mat_sizes[8] = {128, 256, 512, 1028, 2048, 4096, 8192, 16384};
    double cublas_time[8];
    double cublas_gflops[8];
    double xgemm_time[8];
    double xgemm_gflops[8];

    float elapsed_time;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    for (int mat_size = 0; mat_size < 8; mat_size++)
    {
        int Nrows_A = mat_sizes[mat_size];
        int Nrows_B = mat_sizes[mat_size];
        int Ncols_B = mat_sizes[mat_size];

        // Generate random square matrices A, B and C
        float* A = new float[Nrows_A*Nrows_B];
        for (int i = 0; i < Nrows_A; i++)
        {
            for (int j = 0; j < Nrows_B; j++)
            {
                A[i*Nrows_B+j] = (float)(rand() % (MAX_NUM - MIN_NUM + 1) + MIN_NUM);
            }
        }

        float* B = new float[Nrows_B*Ncols_B];
        for (int i = 0; i < Nrows_B; i++)
        {
            for (int j = 0; j < Ncols_B; j++)
            {
                B[i*Ncols_B+j] = (float)(rand() % (MAX_NUM - MIN_NUM + 1) + MIN_NUM);
            }
        }

        float* C_cublas = new float[Nrows_A*Ncols_B];
        float* C_xgemm = new float[Nrows_A*Ncols_B];
        for (int i = 0; i < Nrows_A; i++)
        {
            for (int j = 0; j < Ncols_B; j++)
            {
                C_cublas[i*Ncols_B+j] = 1.0f;
                C_xgemm[i*Ncols_B+j] = -1.0f;
            }
        }

        //-----------------------------------------------------------------------------------------------//
        //------------------------------------- GPU Computations ----------------------------------------//
        //-----------------------------------------------------------------------------------------------//
        // Device array pointers
        float* d_A;
        float* d_B;
        float* d_C_cublas; // Results from CuBLAS
        float* d_C_xgemm; // Results from xGeMM

        // Device memory allocation
        cudaError_t err_A = cudaMalloc((void**) &d_A, Nrows_A*Nrows_B*sizeof(float));
        cuda_check(err_A);

        cudaError_t err_B = cudaMalloc((void**) &d_B, Nrows_B*Ncols_B*sizeof(float));
        cuda_check(err_B);

        cudaError_t err_C_cublas = cudaMalloc((void**) &d_C_cublas, Nrows_A*Ncols_B*sizeof(float));
        cuda_check(err_C_cublas);

        cudaError_t err_C_xgemm = cudaMalloc((void**) &d_C_xgemm, Nrows_A*Ncols_B*sizeof(float));
        cuda_check(err_C_xgemm);

        // Copying A and B to device memory
        cudaError_t err_A_ = cudaMemcpy(d_A, A, Nrows_A*Nrows_B*sizeof(float), cudaMemcpyHostToDevice);
        cuda_check(err_A_);

        cudaError_t err_B_ = cudaMemcpy(d_B, B, Nrows_B*Ncols_B*sizeof(float), cudaMemcpyHostToDevice);
        cuda_check(err_B_);

        //----------------------------------------------------//
        //-------------------- Warmup Run --------------------//
        //----------------------------------------------------//
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
                                d_C_cublas, Ncols_B) // Num cols of C
                    );

        // Tiled Kernel execution
        tiled_xgemm(d_A, d_B, d_C_xgemm, Nrows_A, Nrows_B, Ncols_B, 32, 32);

        // Copying results back
        cudaError_t err_C__cublas = cudaMemcpy(C_cublas, d_C_cublas, Nrows_A*Ncols_B*sizeof(float), cudaMemcpyDeviceToHost);
        cuda_check(err_C__cublas);

        cudaError_t err_C__xgemm = cudaMemcpy(C_xgemm, d_C_xgemm, Nrows_A*Ncols_B*sizeof(float), cudaMemcpyDeviceToHost);
        cuda_check(err_C__xgemm);

        // Asserting Results
        std::cout << "Asserting Results for N: " << Ncols_B << "\n";
        for (int i = 0; i < Nrows_A; i++)
        {
            for (int j = 0; j < Ncols_B; j++)
            {
                if (fabs(C_cublas[i*Ncols_B+j] - C_xgemm[i*Ncols_B+j]) > 0.0001f)
                {
                    std::cerr << "Assertion failed for " << "row number: " << i << ", col number: " << j << ".\n"
                            << "Absolute Difference: " << fabs(C_cublas[i*Ncols_B+j] - C_xgemm[i*Ncols_B+j]) << "\n";
                    assert(fabs(C_cublas[i*Ncols_B+j] - C_xgemm[i*Ncols_B+j]) < 0.0001f);
                }
            }
        }
        std::cout << "Assertion Passed! \n \n";

        //----------------------------------------------------//
        //----------------------- CuBLAS ---------------------//
        //----------------------------------------------------//
        cudaEventRecord(beg);
        for (int n_runs = 0; n_runs < 10; n_runs++)
        {
            cublas_check(cublasSgemm(handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                Ncols_B, Nrows_A, Nrows_B, // Num Cols of C, Num rows of C, Shared dim of A and B
                                &alpha,
                                d_B, Ncols_B, // Num cols of B
                                d_A, Nrows_B, // Num cols of A
                                &beta,
                                d_C_cublas, Ncols_B) // Num cols of C
                    );
        }
        cudaEventRecord(end);
        cudaEventSynchronize(beg);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed_time, beg, end);
        elapsed_time /= 1000.;
        
        cublas_time[mat_size] = elapsed_time / 10;
        cublas_gflops[mat_size] = 2. * 1e-9 * 10 * Nrows_A * Nrows_B * Ncols_B / elapsed_time;

        //----------------------------------------------------//
        //------------------------ xGeMM ---------------------//
        //----------------------------------------------------//
        cudaEventRecord(beg);
        for (int n_runs = 0; n_runs < 10; n_runs++)
        {
            tiled_xgemm(d_A, d_B, d_C_xgemm, Nrows_A, Nrows_B, Ncols_B, 32, 32);
        }
        cudaEventRecord(end);
        cudaEventSynchronize(beg);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed_time, beg, end);
        elapsed_time /= 1000.;
        
        xgemm_time[mat_size] = elapsed_time / 10;
        xgemm_gflops[mat_size] = 2. * 1e-9 * 10 * Nrows_A * Nrows_B * Ncols_B / elapsed_time;


        // Free memory
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C_cublas);
        cudaFree(d_C_xgemm);

        delete[] A;
        delete[] B;
        delete[] C_cublas;
        delete[] C_xgemm;

    }

    std::cout << "CuBLAS Time: ";
    for (int mat_size = 0; mat_size < 8; mat_size++)
        std::cout << cublas_time[mat_size] << " ";
    std::cout << "\n";
    std::cout << "xGeMM Time: ";
    for (int mat_size = 0; mat_size < 8; mat_size++)
        std::cout << xgemm_time[mat_size] << " ";
    std::cout << "\n \n";

    std::cout << "CuBLAS GFLOPS: ";
    for (int mat_size = 0; mat_size < 8; mat_size++)
        std::cout << cublas_gflops[mat_size] << " ";
    std::cout << "\n";
    std::cout << "xGeMM GFLOPS: ";
    for (int mat_size = 0; mat_size < 8; mat_size++)
        std::cout << xgemm_gflops[mat_size] << " ";
    std::cout << "\n \n";

    std::cout << "Speedup with xGeMM against CuBLAS (CuBLAS/xGeMM): ";
    for (int mat_size = 0; mat_size < 8; mat_size++)
        std::cout << cublas_time[mat_size]/xgemm_time[mat_size] << "x ";
    std::cout << "\n";
    

    return 0;
}
