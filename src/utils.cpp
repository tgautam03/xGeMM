#include <iostream>
#include <assert.h>
#include <random>

void matrix_assert(float* A, float* B, int Nrows, int Ncols)
{
    for (int i = 0; i < Nrows; i++)
    {
        for (int j = 0; j < Ncols; j++)
        {
            if (fabs(A[i*Ncols+j] - B[i*Ncols+j]) > 0.0001f)
            {
                std::cerr << "Assertion failed for " << "row number: " << i << ", col number: " << j << ".\n"
                        << "Absolute Difference: " << fabs(A[i*Ncols+j] - B[i*Ncols+j]) << "\n";
                assert(fabs(A[i*Ncols+j] - B[i*Ncols+j]) < 0.000001f);
            }
        }
    }
}