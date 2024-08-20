__global__ void coalesced_mat_mul_kernel(float* d_A, float* d_B, float* d_C, int N1, int N2, int N3)
{
    // Working on C[i,j]
    int i = blockDim.y*blockIdx.y + threadIdx.y;
    int j = blockDim.x*blockIdx.x + threadIdx.x;

    // Parallel mat mul
    if (i < N1 && j < N3)
    {
        // Value at C[i,j]
        float value = 0;
        for (int k = 0; k < N2; k++)
        {
            value += d_A[i*N2+k] * d_B[k*N3+j];
        }

        // Assigning calculated value
        d_C[i*N3+j] = value;
    }
}

void coalesced_xgemm(float* d_A, float* d_B, float* d_C, int Nrows_A, int Nrows_B, int Ncols_B, const int dim_block_x, const int dim_block_y)
{
    // Kernel execution
    dim3 dim_block(dim_block_x, dim_block_y, 1);
    dim3 dim_grid(ceil(Ncols_B/(float)(dim_block_x)), ceil(Nrows_A/(float)(dim_block_y)), 1);
    coalesced_mat_mul_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C, Nrows_A, Nrows_B, Ncols_B);
}