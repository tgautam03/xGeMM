#ifndef UTILS
#define UTILS
#include <iostream>
#include <assert.h>
#include <random>
#include <iomanip>
#include <fstream>
#include "../include/MatrixFP32.cuh"

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

// Initalizing MatrixFP32 with val
void init_mat(MatrixFP32 mat, float val);

// Initalizing MatrixFP32 randomly between (MAX_VAL, MIN_VAL)
void random_init_mat(MatrixFP32 mat, int MAX_VAL, int MIN_VAL);

// Print Matrix (Full or Partial)
void print_mat(MatrixFP32 mat, bool full);

// Asserting matrices are same within the tolerance (eps)
void assert_mat(MatrixFP32 A_mat, MatrixFP32 B_mat, float eps);

// Update benchmark.txt file with recorded times and GFLOPS
void update_benckmark_txt(const std::string& filename, const double recorded_times[], 
                        const double recorded_gflops[], const int mat_sizes[], 
                        const int n_sizes);
#endif