#include <assert.h>
#include <iostream>

#include "../include/MatrixFP32.cuh"
#include "../include/utils.cuh"

// #define cuda_check(err) {if (err != cudaSuccess){printf("%s in %s at line %d \n", cudaGetErrorString(err), __FILE__, __LINE__);exit(EXIT_FAILURE);}}

MatrixFP32::MatrixFP32(int n_rows_, int n_cols_, bool on_device_) : n_rows(n_rows_), n_cols(n_cols_), on_device(on_device_)
{
    if (on_device_ == false)
    {
        // Initialize dynamic array
        ptr = new float[n_rows*n_cols];
    }
    else
    {
        // Allocate device memory
        cudaError_t err = cudaMalloc((void**) &ptr, n_rows*n_cols*sizeof(float));
        cuda_check(err);
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
    cuda_check(err);
}

void MatrixFP32::copy_to_host(MatrixFP32 h_mat)
{
    // Make sure that ptr is on device
    assert(on_device == true && "Matrix must be in device memory");
    assert(h_mat.on_device == false && "Input Matrix to this function must be in host memory");

    // Copying from host to device memory
    cudaError_t err = cudaMemcpy(h_mat.ptr, ptr, n_rows*n_cols*sizeof(float), cudaMemcpyDeviceToHost);
    cuda_check(err);
}