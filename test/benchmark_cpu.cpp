#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <random>
#include <chrono>
#include <Eigen/Dense>

#include "../include/MatrixFP32.cuh"
#include "../include/utils.hpp"
#include "../include/cpu_xgemm.hpp"

int main(int argc, char const *argv[])
{
    // Options: 128, 256, 512, 1028, 2048, 4096, 8192
    int mat_sizes[] = {128, 256, 512, 1028, 2048, 4096};
    int n_sizes = sizeof(mat_sizes) / sizeof(mat_sizes[0]);
    
    double cblas_time[n_sizes];
    double cblas_gflops[n_sizes];

    double cpu_time[n_sizes];
    double cpu_gflops[n_sizes];


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
        MatrixFP32 C_FP32 = MatrixFP32(n, n, false);

        // Initialize Matrices
        random_init_mat(A_FP32, -10, 10); // Random Initialization between -10 and 10
        random_init_mat(B_FP32, -10, 10); // Random Initialization between -10 and 10
        init_mat(C_FP32, -1.0f); // Initialize to -1 (Different from C_eigen)

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
        cpu_xgemm(A_FP32, B_FP32, C_FP32);
        // Assert Results
        double eps = 1e-8;
        std::cout << "Asserting... " << "n: " << n << "\n";
        for (int i = 0; i < C_FP32.n_rows; i++)
        {
            for (int j = 0; j < C_FP32.n_cols; j++)
            {
                if (fabs(C_FP32.ptr[i*C_FP32.n_cols + j] - C_eigen(i,j)) > eps)
                {
                    std::cerr << "Assertion failed for " << "row number: " << i << ", col number: " << j << ".\n"
                            << "Absolute Difference: " << fabs(C_FP32.ptr[i*C_FP32.n_cols + j] - C_eigen(i,j)) << "\n";
                    assert(fabs(C_FP32.ptr[i*C_FP32.n_cols + j] - C_eigen(i,j)) < eps && "Assertion failed!");
                }
            }
        }
        std::cout << "Passed! \n \n";

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

        //----------------------------------------------------//
        //----------------------- CPU ------------------------//
        //----------------------------------------------------//
        start = std::chrono::high_resolution_clock::now();
        for (int n_runs = 0; n_runs < 10; n_runs++)
            cpu_xgemm(A_FP32, B_FP32, C_FP32);
        stop = std::chrono::high_resolution_clock::now();

        elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        cpu_time[mat_size] = (elapsed_time.count()/1e+6) / 10;
        cpu_gflops[mat_size] = 2. * 1e-9 * 10 * n * n * n / (elapsed_time.count()/1e+6);

        // Free xGeMM Memory
        A_FP32.free_mat();
        B_FP32.free_mat();
        C_FP32.free_mat();
    }

    std::cout << "Matrix Size: ";
    for (int mat_size = 0; mat_size < n_sizes; mat_size++)
        std::cout << mat_sizes[mat_size] << " ";
    std::cout << "\n \n";

    std::cout << "cBLAS Time (seconds): ";
    for (int mat_size = 0; mat_size < n_sizes; mat_size++)
        std::cout << cblas_time[mat_size] << " ";
    std::cout << "\n";
    std::cout << "CPU Time (seconds): ";
    for (int mat_size = 0; mat_size < n_sizes; mat_size++)
        std::cout << cpu_time[mat_size] << " ";
    std::cout << "\n \n";

    std::cout << "cBLAS GFLOPS: ";
    for (int mat_size = 0; mat_size < n_sizes; mat_size++)
        std::cout << cblas_gflops[mat_size] << " ";
    std::cout << "\n";
    std::cout << "CPU GFLOPS: ";
    for (int mat_size = 0; mat_size < n_sizes; mat_size++)
        std::cout << cpu_gflops[mat_size] << " ";
    std::cout << "\n \n";

    std::cout << "BLAS vs xGeMM (BLAS/xGeMM): ";
    for (int mat_size = 0; mat_size < n_sizes; mat_size++)
        std::cout << std::fixed << std::setprecision(2) << cblas_time[mat_size]/cpu_time[mat_size]*100 << "% ";
    std::cout << "\n";

    // Saving to benchmark file
    update_benckmark_txt("txt_benchmarks/cpu.txt", cpu_time, cpu_gflops, mat_sizes, n_sizes);
    update_benckmark_txt("txt_benchmarks/blas.txt", cblas_time, cblas_gflops, mat_sizes, n_sizes);
    return 0;
}
