#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <random>

#include <cublas_v2.h>

#include "../include/cublas_sgemm.cuh"
#include "../include/tiled_xgemm.cuh"
#include "../include/utils.hpp"

#define MAX_NUM 10 
#define MIN_NUM -10

int main(int argc, char const *argv[])
{
    int Nrows_A = 10723;
    int Nrows_B = 10674;
    int Ncols_B = 11765;

    // Generate NxN square matrices A and B
    // Create a random device and seed the generator
    std::random_device rd;
    std::mt19937 gen(rd());

    // Define the range for the random float numbers
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    float* A = new float[Nrows_A*Nrows_B];
    for (int i = 0; i < Nrows_A; i++)
    {
        for (int j = 0; j < Nrows_B; j++)
        {
            A[i*Nrows_B+j] = dist(gen);
        }
    }

    float* B = new float[Nrows_B*Ncols_B];
    for (int i = 0; i < Nrows_B; i++)
    {
        for (int j = 0; j < Ncols_B; j++)
        {
            B[i*Ncols_B+j] = dist(gen);
        }
    }
    

    // CUBLAS SGEMM
    float* C = new float[Nrows_A*Ncols_B];
    unsigned long long cublas_sgemm_t1_gpu = myCPUTimer();
    cublas_sgemm(A, B, C, Nrows_A, Nrows_B, Ncols_B);
    unsigned long long cublas_sgemm_t2_gpu = myCPUTimer();
    std::cout << "CUBLAS SGEMM execution time: " << cublas_sgemm_t2_gpu-cublas_sgemm_t1_gpu << " microseconds \n";

    // CUBLAS SGEMM
    float* xC = new float[Nrows_A*Ncols_B];
    unsigned long long x_t1_gpu = myCPUTimer();
    tiled_xgemm(A, B, xC, Nrows_A, Nrows_B, Ncols_B);
    unsigned long long x_t2_gpu = myCPUTimer();
    std::cout << "Tiled xGEMM execution time: " << x_t2_gpu-x_t1_gpu << " microseconds \n";

    // // CUBLAS
    // std::cout << "C Matrix \n";
    // for (int i = 0; i < Nrows_A; i++)
    // {
    //     for (int j = 0; j < Ncols_B; j++)
    //         std::cout << C[i*Ncols_B+j] << " ";
    //     std::cout << "\n";
    // }

    // // xGeMM
    // std::cout << "xC Matrix \n";
    // for (int i = 0; i < Nrows_A; i++)
    // {
    //     for (int j = 0; j < Ncols_B; j++)
    //         std::cout << xC[i*Ncols_B+j] << " ";
    //     std::cout << "\n";
    // }

    // Asserting Results
    std::cout << "Asserting Results... \n";
    for (int i = 0; i < Nrows_A; i++)
    {
        for (int j = 0; j < Ncols_B; j++)
        {
            if (fabs(C[i*Ncols_B+j] - xC[i*Ncols_B+j]) > 0.0001f)
            {
                std::cerr << "Assertion failed for " << "row number: " << i << ", col number: " << j << ".\n"
                        << "Absolute Difference: " << fabs(C[i*Ncols_B+j] - xC[i*Ncols_B+j]) << "\n";
                assert(fabs(C[i*Ncols_B+j] - xC[i*Ncols_B+j]) < 0.0001f);
            }
        }
    }
    std::cout << "Asserting Passed! \n";


    // Free memory
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] xC;

    return 0;
}
