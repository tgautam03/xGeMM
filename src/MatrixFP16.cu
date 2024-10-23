#include <assert.h>
#include <iostream>

#include "../include/MatrixFP16.cuh"
#include "../include/utils.cuh"

// #define cuda_check(err) {if (err != cudaSuccess){printf("%s in %s at line %d \n", cudaGetErrorString(err), __FILE__, __LINE__);exit(EXIT_FAILURE);}}

MatrixFP16::MatrixFP16(int n_rows_, int n_cols_, bool on_device_) : n_rows(n_rows_), n_cols(n_cols_), on_device(on_device_)
{
    if (on_device_ == false)
    {
        // Initialize dynamic array
        ptr = new half[n_rows*n_cols];
    }
    else
    {
        // Allocate device memory
        cudaError_t err = cudaMalloc((void**) &ptr, n_rows*n_cols*sizeof(half));
        cuda_check(err);
    }
}

void MatrixFP16::free_mat()
{
    if (on_device == false)
        delete[] ptr;
    else
        cudaFree(ptr);
}

void MatrixFP16::copy_to_device(MatrixFP16 d_mat)
{
    // Make sure that ptr is on host 
    assert(on_device == false && "Matrix must be in host memory");
    assert(d_mat.on_device == true && "Input Matrix to this function must be in device memory");

    // Copying from host to device memory
    cudaError_t err = cudaMemcpy(d_mat.ptr, ptr, n_rows*n_cols*sizeof(half), cudaMemcpyHostToDevice);
    cuda_check(err);
}

void MatrixFP16::copy_to_host(MatrixFP16 h_mat)
{
    // Make sure that ptr is on device
    assert(on_device == true && "Matrix must be in device memory");
    assert(h_mat.on_device == false && "Input Matrix to this function must be in host memory");

    // Copying from host to device memory
    cudaError_t err = cudaMemcpy(h_mat.ptr, ptr, n_rows*n_cols*sizeof(half), cudaMemcpyDeviceToHost);
    cuda_check(err);
}