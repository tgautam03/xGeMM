#include <assert.h>
#include <iostream>

#include "../include/MatrixFP32.cuh"

#define CUDA_CHECK(err) {if (err != cudaSuccess){printf("%s in %s at line %d \n", cudaGetErrorString(err), __FILE__, __LINE__);exit(EXIT_FAILURE);}}

MatrixFP32::MatrixFP32(int n_rows_, int n_cols_, bool on_device_)
{
    // Assigning Number of rows and cols to provate variables
    n_rows = n_rows_;
    n_cols = n_cols_;

    if (on_device_ == false)
    {
        // Initialize dynamic array
        ptr = new float[n_rows*n_cols];
        // Matrix is in host memory (RAM)
        on_device = on_device_;
    }
    else
    {
        // Allocate device memory
        cudaError_t err = cudaMalloc((void**) &ptr, n_rows*n_cols*sizeof(float));
        CUDA_CHECK(err);
        // Matrix is in device memory (VRAM)
        on_device = on_device_;
    }
}

void MatrixFP32::free_mat()
{
    if (on_device == false)
        delete[] ptr;
    else
        cudaFree(ptr);
}

void MatrixFP32::copy_to_device(MatrixFP32 d_mat)
{
    // Make sure that ptr is on host 
    assert(on_device == false && "Matrix must be in host memory");
    assert(d_mat.on_device == true && "Input Matrix to this function must be in device memory");

    // Copying from host to device memory
    cudaError_t err = cudaMemcpy(d_mat.ptr, ptr, n_rows*n_cols*sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err);
}

void MatrixFP32::copy_to_host(MatrixFP32 h_mat)
{
    // Make sure that ptr is on device
    assert(on_device == true && "Matrix must be in device memory");
    assert(h_mat.on_device == false && "Input Matrix to this function must be in host memory");

    // Copying from host to device memory
    cudaError_t err = cudaMemcpy(h_mat.ptr, ptr, n_rows*n_cols*sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK(err);
}