#include <iostream>
#include <assert.h>
#include <random>
#include <iomanip>
#include <fstream>
#include <Eigen/Dense>
#include "../include/MatrixFP32.cuh"

#define CUDA_CHECK(err) {if (err != cudaSuccess){printf("%s in %s at line %d \n", cudaGetErrorString(err), __FILE__, __LINE__);exit(EXIT_FAILURE);}}

void init_mat(MatrixFP32 mat, float val)
{
    // Getting Matrix Dimension
    int n_rows = mat.rows(); 
    int n_cols = mat.cols();

    // Initializing val to each location
    for (int i = 0; i < n_rows; i++)
    {
        for (int j = 0; j < n_cols; j++)
        {
            mat.set_val(i, j, val);
        }
    }
}

void init_mat(MatrixFP32 mat, int MAX_VAL, int MIN_VAL)
{
    // Getting Matrix Dimension
    int n_rows = mat.rows(); 
    int n_cols = mat.cols();

    // Initializing val to each location
    for (int i = 0; i < n_rows; i++)
    {
        for (int j = 0; j < n_cols; j++)
        {
            mat.set_val(i, j, (float)(rand() % (MAX_VAL - MIN_VAL + 1) + MIN_VAL));
        }
    }
}

void print_mat(MatrixFP32 mat, bool full)
{
    // Getting Matrix Dimension
    int n_rows = mat.rows(); 
    int n_cols = mat.cols();

    if (full == true)
    {
        // Print Full Matrix
        for (int i = 0; i < n_rows; i++)
        {
            std::cout << "| ";
            for (int j = 0; j < n_cols; j++)
            {
                std::cout << std::setw(5) << std::to_string(mat.get_val(i, j)) << " ";
            }
            std::cout << " |" << "\n";
        }
    }
    else
    {
        // Print Partial Matrix
        for (int i = 0; i < 5; i++)
        {
            std::cout << "| ";
            for (int j = 0; j < 5; j++)
            {
                if (i == 3)
                {
                    std::cout << std::setw(5) << "  ...   " << " ";
                }
                else
                {
                    if (j != 3)
                        std::cout << std::setw(5) << std::to_string(mat.get_val(i, j)) << " ";
                    else
                        std::cout << std::setw(5) << "..." << " ";
                }
            }
            std::cout << " |" << "\n";

        }
    }
}

void print_mat(Eigen::MatrixXd mat, bool full)
{
    // Getting Matrix Dimension
    int n_rows = mat.rows(); 
    int n_cols = mat.cols();

    if (full == true)
    {
        // Print Full Matrix
        for (int i = 0; i < n_rows; i++)
        {
            std::cout << "| ";
            for (int j = 0; j < n_cols; j++)
            {
                std::cout << std::setw(5) << std::to_string(mat(i, j)) << " ";
            }
            std::cout << " |" << "\n";
        }
    }
    else
    {
        // Print Partial Matrix
        for (int i = 0; i < 5; i++)
        {
            std::cout << "| ";
            for (int j = 0; j < 5; j++)
            {
                if (i == 3)
                {
                    std::cout << std::setw(5) << "  ...   " << " ";
                }
                else
                {
                    if (j != 3)
                        std::cout << std::setw(5) << std::to_string(mat(i, j)) << " ";
                    else
                        std::cout << std::setw(5) << "..." << " ";
                }
            }
            std::cout << " |" << "\n";

        }
    }
}

void assert_mat(MatrixFP32 A_mat, MatrixFP32 B_mat, float eps)
{
    // Getting A Matrix Dimension
    int A_n_rows = A_mat.rows(); 
    int A_n_cols = A_mat.cols();

    // Getting B Matrix Dimension
    int B_n_rows = B_mat.rows(); 
    int B_n_cols = B_mat.cols();

    // Asserting that matrices have same dimensions
    assert (A_n_rows == B_n_rows && "A rows must be equal to B rows");
    assert (A_n_cols == B_n_cols && "A cols must be equal to B cols");

    for (int i = 0; i < A_n_rows; i++)
    {
        for (int j = 0; j < A_n_cols; j++)
        {
            if (fabs(A_mat.get_val(i, j) - B_mat.get_val(i,j)) > eps)
            {
                std::cerr << "Assertion failed for " << "row number: " << i << ", col number: " << j << ".\n"
                        << "Absolute Difference: " << fabs(A_mat.get_val(i,j) - B_mat.get_val(i,j)) << "\n";
                assert(fabs(A_mat.get_val(i,j) - B_mat.get_val(i,j)) < eps && "Assertion failed!");
            }
        }
    }
}

void assert_mat(MatrixFP32 A_mat, Eigen::MatrixXd B_mat, float eps)
{
    // Getting A Matrix Dimension
    int A_n_rows = A_mat.rows(); 
    int A_n_cols = A_mat.cols();

    // Getting B Matrix Dimension
    int B_n_rows = B_mat.rows(); 
    int B_n_cols = B_mat.cols();
    

    // Asserting that matrices have same dimensions
    assert (A_n_rows == B_n_rows && "A rows must be equal to B rows");
    assert (A_n_cols == B_n_cols && "A cols must be equal to B cols");

    for (int i = 0; i < A_n_rows; i++)
    {
        for (int j = 0; j < A_n_cols; j++)
        {
            if (fabs(A_mat.get_val(i, j) - B_mat(i,j)) > eps)
            {
                std::cerr << "Assertion failed for " << "row number: " << i << ", col number: " << j << ".\n"
                        << "Absolute Difference: " << fabs(A_mat.get_val(i,j) - B_mat(i,j)) << "\n";
                assert(fabs(A_mat.get_val(i,j) - B_mat(i,j)) < eps && "Assertion failed!");
            }
        }
    }
}

void update_benckmark_txt(const std::string& filename, const double recorded_times[], 
                        const double recorded_gflops[], const int mat_sizes[], 
                        const int n_sizes)
{
    // Opening File
    std::ofstream file(filename);
    if (!file.is_open()) 
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    file << "Matrix Sizes" << ": ";
    for (int i = 0; i < n_sizes; i++)
    {
        if (i != n_sizes-1)
            file << mat_sizes[i] << " " ;
        else
            file << mat_sizes[i] << "\n \n" ;
    }

    file << "Time (Seconds)" << ": ";
    for (int i = 0; i < n_sizes; i++)
    {
        if (i != n_sizes-1)
            file << recorded_times[i] << " " ;
        else
            file << recorded_times[i] << "\n \n" ;
    }

    file << "GPLOPS" << ": ";
    for (int i = 0; i < n_sizes; i++)
    {
        if (i != n_sizes-1)
            file << recorded_gflops[i] << " " ;
        else
            file << recorded_gflops[i] << "\n \n" ;
    }
    
}

// void copy_host_to_device(MatrixFP32 h_mat, MatrixFP32 d_mat)
// {
//     // Make sure that h_mat is on host and d_mat is on device
//     assert(h_mat.on_device_ == false && "Matrix must be in host memory");
//     assert(d_mat.on_device_ == true && "Matrix must be in device memory");

//     // Getting host Matrix Dimension
//     int h_n_rows = h_mat.rows(); 
//     int h_n_cols = h_mat.cols();

//     // Getting device Matrix Dimension
//     int d_n_rows = d_mat.rows(); 
//     int d_n_cols = d_mat.cols();

//     // Asserting that matrices have same dimensions
//     assert (h_n_rows == d_n_rows && "Host rows must be equal to device rows");
//     assert (h_n_cols == d_n_cols && "Host cols must be equal to device cols");

//     // Copying from host to device memory
//     cudaError_t err = cudaMemcpy(d_mat._mat, h_mat._mat, h_n_rows*h_n_cols*sizeof(float), cudaMemcpyHostToDevice);
//     CUDA_CHECK(err);
// }

// void copy_device_to_host(MatrixFP32 d_mat, MatrixFP32 h_mat)
// {
//     // Make sure that h_mat is on host and d_mat is on device
//     assert(h_mat.on_device_ == false && "Matrix must be in host memory");
//     assert(d_mat.on_device_ == true && "Matrix must be in device memory");

//     // Getting host Matrix Dimension
//     int h_n_rows = h_mat.rows(); 
//     int h_n_cols = h_mat.cols();

//     // Getting device Matrix Dimension
//     int d_n_rows = d_mat.rows(); 
//     int d_n_cols = d_mat.cols();

//     // Asserting that matrices have same dimensions
//     assert (h_n_rows == d_n_rows && "Host rows must be equal to device rows");
//     assert (h_n_cols == d_n_cols && "Host cols must be equal to device cols");

//     // Copying from host to device memory
//     cudaError_t err = cudaMemcpy(h_mat._mat, d_mat._mat, h_n_rows*h_n_cols*sizeof(float), cudaMemcpyHostToDevice);
//     CUDA_CHECK(err);
// }