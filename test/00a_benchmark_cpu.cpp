#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <random>
#include <chrono>
#include <Eigen/Dense>

#include "../include/MatrixFP32.cuh"
#include "../include/utils.cuh"

int main(int argc, char const *argv[])
{
    // Options: Anything!
    int mat_sizes[] = {128, 256, 512, 1024, 2048, 4096};
    int n_sizes = sizeof(mat_sizes) / sizeof(mat_sizes[0]);
    
    double cblas_time[n_sizes];
    double cblas_gflops[n_sizes];

    auto start = std::chrono::high_resolution_clock::now();
    auto stop = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    for (int mat_size = 0; mat_size < n_sizes; mat_size++)
    {
        // Matrix Size
        int n = mat_sizes[mat_size];

        // Define MatrixFP32
        MatrixFP32 A_FP32 = MatrixFP32(n, n, false);
        MatrixFP32 B_FP32 = MatrixFP32(n, n, false);

        // Initialize Matrices
        random_init_mat(A_FP32, -10, 10); // Random Initialization between -10 and 10
        random_init_mat(B_FP32, -10, 10); // Random Initialization between -10 and 10

        // Generate Eigen square matrices A, B and C
        Eigen::MatrixXd A_eigen(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
                A_eigen(i,j) = A_FP32.ptr[i*A_FP32.n_cols + j];
        }

        Eigen::MatrixXd B_eigen(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
                B_eigen(i,j) = B_FP32.ptr[i*B_FP32.n_cols + j];
        }

        Eigen::MatrixXd C_eigen(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
                C_eigen(i,j) = 0.0f;
        }

        //----------------------------------------------------//
        //-------------------- Warmup Run --------------------//
        //----------------------------------------------------//
        // Perform matrix multiplication: C = A * B 
        C_eigen = A_eigen * B_eigen;

        //----------------------------------------------------//
        //---------------------- cBLAS -----------------------//
        //----------------------------------------------------//
        start = std::chrono::high_resolution_clock::now();
        for (int n_runs = 0; n_runs < 10; n_runs++)
            C_eigen = A_eigen * B_eigen;
        stop = std::chrono::high_resolution_clock::now();

        elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        cblas_time[mat_size] = (elapsed_time.count()/1e+6) / 10;
        cblas_gflops[mat_size] = 2. * 1e-9 * 10 * n * n * n / (elapsed_time.count()/1e+6);

        // Free xGeMM Memory
        A_FP32.free_mat();
        B_FP32.free_mat();
    }

    std::cout << "Matrix Size: ";
    for (int mat_size = 0; mat_size < n_sizes; mat_size++)
        std::cout << mat_sizes[mat_size] << " ";
    std::cout << "\n \n";

    std::cout << "cBLAS Time (seconds): ";
    for (int mat_size = 0; mat_size < n_sizes; mat_size++)
        std::cout << cblas_time[mat_size] << " ";
    std::cout << "\n \n";

    std::cout << "cBLAS GFLOPS: ";
    for (int mat_size = 0; mat_size < n_sizes; mat_size++)
        std::cout << cblas_gflops[mat_size] << " ";
    std::cout << "\n \n";

    // Saving to benchmark file
    update_benckmark_txt("txt_benchmarks/00a_blas.txt", cblas_time, cblas_gflops, mat_sizes, n_sizes);
    return 0;
}
