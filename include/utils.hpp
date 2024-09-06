#ifndef UTILS
#define UTILS
#include <iostream>
#include <assert.h>
#include <random>
#include <iomanip>
#include <fstream>
#include "../include/MatrixFP32.cuh"

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