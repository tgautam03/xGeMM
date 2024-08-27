#include <assert.h>
#include <iostream>

#include "../include/MatrixFP32.cuh"

#define CUDA_CHECK(err) {if (err != cudaSuccess){printf("%s in %s at line %d \n", cudaGetErrorString(err), __FILE__, __LINE__);exit(EXIT_FAILURE);}}

MatrixFP32::MatrixFP32(int n_rows, int n_cols, bool on_device)
{
    // Assigning Number of rows and cols to provate variables
    _n_rows = n_rows;
    _n_cols = n_cols;

    if (on_device == false)
    {
        // Initialize dynamic array
        _mat = new float[_n_rows*_n_cols];
        // Matrix is in host memory (RAM)
        _on_device = on_device;
    }
    else
    {
        // Allocate device memory
        cudaError_t err = cudaMalloc((void**) &_mat, n_rows*n_cols*sizeof(float));
        CUDA_CHECK(err);
        // Matrix is in device memory (VRAM)
        _on_device = on_device;
    }
}

void MatrixFP32::free_mat()
{
    if (_on_device == false)
        delete[] _mat;
    else
        cudaFree(_mat);
}

int MatrixFP32::rows()
{
    return _n_rows;
}

int MatrixFP32::cols()
{
    return _n_cols;
}

float MatrixFP32::get_val(int row, int col)
{
    return _mat[row*_n_cols + col];
}

void MatrixFP32::set_val(int row, int col, float val)
{
    _mat[row*_n_cols + col] = val;
}

MatrixFP32 MatrixFP32::copy_to_device()
{
    // Make sure that _mat is on host 
    assert(_on_device == false && "Matrix must be in host memory");

    // Initialize Device Matrix
    MatrixFP32 d_mat(_n_rows, _n_cols, true);

    // Copying from host to device memory
    cudaError_t err = cudaMemcpy(d_mat._mat, _mat, _n_rows*_n_cols*sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err);

    return d_mat;
}

MatrixFP32 MatrixFP32::copy_to_host()
{
    // Make sure that _mat is on device
    assert(_on_device == true && "Matrix must be in host memory");

    // Initialize Device Matrix
    MatrixFP32 h_mat(_n_rows, _n_cols, false);

    // Copying from host to device memory
    cudaError_t err = cudaMemcpy(h_mat._mat, _mat, _n_rows*_n_cols*sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK(err);

    return h_mat;
}

