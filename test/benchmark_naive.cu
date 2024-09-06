#include <iostream>
#include <iomanip>
#include <cublas_v2.h>

#include "../include/MatrixFP32.cuh"
#include "../include/utils.hpp"

#include "../include/naive_xgemm.cuh"

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
    // Options: 8, 16, 32, 64, 128, 256, 512, 1028, 2048, 4096, 8192
    int mat_sizes[] = {128, 256, 512, 1028, 2048, 4096};
    int n_sizes = sizeof(mat_sizes) / sizeof(mat_sizes[0]);

    // For recording time
    float elapsed_time;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    // Store time and GFLOPS
    double cublas_time[n_sizes];
    double cublas_gflops[n_sizes];
    double xgemm_time[n_sizes];
    double xgemm_gflops[n_sizes];

    for (int mat_size = 0; mat_size < n_sizes; mat_size++)
    {
        // Matrix Size
        int n = mat_sizes[mat_size];

        // Define MatrixFP32
        MatrixFP32 A_FP32 = MatrixFP32(n, n, false);
        MatrixFP32 B_FP32 = MatrixFP32(n, n, false);
        MatrixFP32 C_FP32_cublas = MatrixFP32(n, n, false);
        MatrixFP32 C_FP32_xgemm = MatrixFP32(n, n, false);

        // Initialize Matrices
        random_init_mat(A_FP32, -10, 10);          // Random Initialization between -10 and 10
        random_init_mat(B_FP32, -10, 10);          // Random Initialization between -10 and 10
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
                                d_C_FP32_cublas.n_cols, d_C_FP32_cublas.n_rows, d_A_FP32.n_cols, // Num Cols of C, Num rows of C, Shared dim of A and B
                                &alpha,
                                d_B_FP32.ptr, d_B_FP32.n_cols, // Num cols of B
                                d_A_FP32.ptr, d_A_FP32.n_cols, // Num cols of A
                                &beta,
                                d_C_FP32_cublas.ptr, d_C_FP32_cublas.n_cols) // Num cols of C
                    );
        cudaDeviceSynchronize();

        // Naive Kernel execution
        naive_xgemm(d_A_FP32, d_B_FP32, d_C_FP32_xgemm);
        cudaDeviceSynchronize();

        // Assert that naive implementation is correct
        d_C_FP32_cublas.copy_to_host(C_FP32_cublas);
        d_C_FP32_xgemm.copy_to_host(C_FP32_xgemm);
        std::cout << "Asserting Results for N: " << n << "\n";
        assert_mat(C_FP32_xgemm, C_FP32_cublas, 1e-8);
        std::cout << "Assertion Passed! \n \n";

        // Printing the smallest matrix result
        if (n == 8)
        {
            std::cout << "Matrix C (cuBLAS): \n";
            print_mat(C_FP32_cublas, true);
            std::cout << "\n";

            std::cout << "Matrix C (xGeMM): \n";
            print_mat(C_FP32_xgemm, true);
            std::cout << "\n";
        }

        //----------------------------------------------------//
        //--------------------- cuBLAS -----------------------//
        //----------------------------------------------------//
        cudaEventRecord(beg);
        for (int n_runs = 0; n_runs < 10; n_runs++)
        {
            float alpha = 1;
            float beta = 0;
            cublas_check(cublasSgemm(handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                d_C_FP32_cublas.n_cols, d_C_FP32_cublas.n_rows, d_A_FP32.n_cols, // Num Cols of C, Num rows of C, Shared dim of A and B
                                &alpha,
                                d_B_FP32.ptr, d_B_FP32.n_cols, // Num cols of B
                                d_A_FP32.ptr, d_A_FP32.n_cols, // Num cols of A
                                &beta,
                                d_C_FP32_cublas.ptr, d_C_FP32_cublas.n_cols) // Num cols of C
                    );
            cudaDeviceSynchronize();
        }
        cudaEventRecord(end);
        cudaEventSynchronize(beg);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed_time, beg, end);
        elapsed_time /= 1000.;

        cublas_time[mat_size] = (elapsed_time) / 10;
        cublas_gflops[mat_size] = 2. * 1e-9 * 10 * n * n * n / (elapsed_time);

        //----------------------------------------------------//
        //---------------------- xGeMM -----------------------//
        //----------------------------------------------------//
        cudaEventRecord(beg);
        for (int n_runs = 0; n_runs < 10; n_runs++)
        {
            naive_xgemm(d_A_FP32, d_B_FP32, d_C_FP32_xgemm);
            cudaDeviceSynchronize();
        }
        cudaEventRecord(end);
        cudaEventSynchronize(beg);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed_time, beg, end);
        elapsed_time /= 1000.;

        xgemm_time[mat_size] = (elapsed_time) / 10;
        xgemm_gflops[mat_size] = 2. * 1e-9 * 10 * n * n * n / (elapsed_time);

        // Free Memory
        A_FP32.free_mat();
        B_FP32.free_mat();
        C_FP32_cublas.free_mat();
        C_FP32_xgemm.free_mat();

        d_A_FP32.free_mat();
        d_B_FP32.free_mat();
        d_C_FP32_cublas.free_mat();
        d_C_FP32_xgemm.free_mat();
    }

    std::cout << "Matrix Size: ";
    for (int mat_size = 0; mat_size < n_sizes; mat_size++)
        std::cout << mat_sizes[mat_size] << " ";
    std::cout << "\n \n";

    std::cout << "cuBLAS Time (seconds): ";
    for (int mat_size = 0; mat_size < n_sizes; mat_size++)
        std::cout << cublas_time[mat_size] << " ";
    std::cout << "\n";
    std::cout << "xGeMM Time (seconds): ";
    for (int mat_size = 0; mat_size < n_sizes; mat_size++)
        std::cout << xgemm_time[mat_size] << " ";
    std::cout << "\n \n";

    std::cout << "cuBLAS GFLOPS: ";
    for (int mat_size = 0; mat_size < n_sizes; mat_size++)
        std::cout << cublas_gflops[mat_size] << " ";
    std::cout << "\n";
    std::cout << "xGeMM GFLOPS: ";
    for (int mat_size = 0; mat_size < n_sizes; mat_size++)
        std::cout << xgemm_gflops[mat_size] << " ";
    std::cout << "\n \n";

    std::cout << "cuBLAS vs Naive xGeMM (CuBLAS/xGeMM): ";
    for (int mat_size = 0; mat_size < n_sizes; mat_size++)
        std::cout << std::fixed << std::setprecision(2) << cublas_time[mat_size]/xgemm_time[mat_size]*100 << "% ";
    std::cout << "\n";

    // Saving to benchmark file
    update_benckmark_txt("txt_benchmarks/cublas.txt", cublas_time, cublas_gflops, mat_sizes, n_sizes);
    update_benckmark_txt("txt_benchmarks/naive_xgemm.txt", xgemm_time, xgemm_gflops, mat_sizes, n_sizes);

    return 0;
}
