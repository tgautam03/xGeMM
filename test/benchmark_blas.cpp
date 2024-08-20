#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <random>
#include <chrono>
#include <Eigen/Dense>

// #include "../include/utils.hpp"


#define MAX_NUM 10 
#define MIN_NUM -10

int main(int argc, char const *argv[])
{
    int mat_sizes[8] = {128, 256, 512, 1028, 2048, 4096, 8192, 16384};
    double cblas_time[8];
    double cblas_gflops[8];

    auto start = std::chrono::high_resolution_clock::now();
    auto stop = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    for (int mat_size = 0; mat_size < 8; mat_size++)
    {
        int Nrows_A = mat_sizes[mat_size];
        int Nrows_B = mat_sizes[mat_size];
        int Ncols_B = mat_sizes[mat_size];

        // Generate random square matrices A, B and C
        Eigen::MatrixXd A(Nrows_A, Nrows_B);
        for (int i = 0; i < Nrows_A; i++)
        {
            for (int j = 0; j < Nrows_B; j++)
            {
                A(i,j) = (float)(rand() % (MAX_NUM - MIN_NUM + 1) + MIN_NUM);
            }
        }

        Eigen::MatrixXd B(Nrows_B, Ncols_B);
        for (int i = 0; i < Nrows_B; i++)
        {
            for (int j = 0; j < Ncols_B; j++)
            {
                B(i,j) = (float)(rand() % (MAX_NUM - MIN_NUM + 1) + MIN_NUM);
            }
        }

        Eigen::MatrixXd C(Nrows_A, Ncols_B);
        for (int i = 0; i < Nrows_A; i++)
        {
            for (int j = 0; j < Ncols_B; j++)
            {
                C(i,j) = 0.0f;
            }
        }

        //----------------------------------------------------//
        //-------------------- Warmup Run --------------------//
        //----------------------------------------------------//
        // Perform matrix multiplication: C = A * B 
        C = A * B;

        //----------------------------------------------------//
        //---------------------- cBLAS -----------------------//
        //----------------------------------------------------//
        start = std::chrono::high_resolution_clock::now();
        for (int n_runs = 0; n_runs < 10; n_runs++)
        {
            C = A * B;
        }
        stop = std::chrono::high_resolution_clock::now();

        elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        cblas_time[mat_size] = (elapsed_time.count()/1e+6) / 10;
        cblas_gflops[mat_size] = 2. * 1e-9 * 10 * Nrows_A * Nrows_B * Ncols_B / (elapsed_time.count()/1e+6);
    }

    std::cout << "cBLAS Time: ";
    for (int mat_size = 0; mat_size < 8; mat_size++)
        std::cout << cblas_time[mat_size] << " ";
    std::cout << "\n \n";

    std::cout << "cBLAS GFLOPS: ";
    for (int mat_size = 0; mat_size < 8; mat_size++)
        std::cout << cblas_gflops[mat_size] << " ";
    std::cout << "\n";

    return 0;
}
