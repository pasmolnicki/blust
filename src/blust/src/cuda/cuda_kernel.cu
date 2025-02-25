#include <iostream>
#include <math.h>
#include <stdio.h>

typedef float cu_number_t;
typedef cu_number_t* pointer_t;


// VECTOR OPERATIONS

// Should be launched with maximum of 32 * 256 threads (obviously based on the data size)
// with 1d parameters (Nbocks, Block size)
extern "C" __global__ void cu_vector_add(pointer_t m1, pointer_t m2, pointer_t res, size_t N)
{
    size_t i      = threadIdx.x + blockDim.x * blockIdx.x;
    size_t stripe = blockDim.x * gridDim.x;

    for (; i < N; i += stripe)
        res[i] = m1[i] + m2[i];
}

// Should be launched with 1d parameters (Cij = M1ij - M2ij)
extern "C" __global__ void cu_vector_sub(pointer_t m1, pointer_t m2, pointer_t res, size_t N)
{
    size_t i      = threadIdx.x + blockDim.x * blockIdx.x;
    size_t stripe = blockDim.x * gridDim.x;

#pragma unroll
    for (; i < N; i+=stripe)
        res[i] = m1[i] - m2[i];
}

// Multiply a vector by a scalar, and store the result in `res` (Bij = Aij * scalar)
extern "C" __global__ void cu_vector_mul_scalar(pointer_t m, cu_number_t scalar, pointer_t res, size_t N)
{
	size_t i = threadIdx.x + blockDim.x * blockIdx.x;
	size_t stripe = blockDim.x * gridDim.x;

	for (; i < N; i += stripe)
		res[i] = m[i] * scalar;
}

// Multiplies A * B, such that Cij = Aij * Bij
extern "C" __global__ void cu_vector_mul_hadamard(pointer_t m1, pointer_t m2, size_t N)
{
    size_t i      = threadIdx.x + blockDim.x * blockIdx.x;
    size_t stripe = blockDim.x * gridDim.x;

    for (; i < N; i+=stripe)
        m1[i] *= m2[i];
}


// MATRIX OPERATIONS

// Transpose the matrix `m` (with rows x cols dimensions), and store the result in `res` (with cols x rows dim)
extern "C" __global__ void cu_mat_transpose(pointer_t m, pointer_t res, size_t rows, size_t cols)
{
    const size_t stripe = blockDim.x * gridDim.x;
    const size_t N      = rows * cols;

    for (size_t n = blockDim.x * blockIdx.x + threadIdx.x; n < N; n += stripe)
    {
        size_t i = n / rows;
        size_t j = n % rows;
        res[n] = m[cols*j + i];
    }
}

// Multiply two matrices, and store the result in `res`
// Should be called with 2d parameters (Nbocks, Block size)
extern "C" __global__ void cu_mat_mul(pointer_t m1, pointer_t m2, pointer_t res, size_t m1_rows, size_t m2_cols, size_t m2_rows)
{
	size_t target_rows = m1_rows;
	size_t target_cols = m2_cols;

	// Get the row and column of the current thread
	size_t row = blockIdx.y * blockDim.y + threadIdx.y;
	size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    size_t stripe_x = blockDim.x * gridDim.x;
    size_t stripe_y = blockDim.y * gridDim.y;

	// Check if the current thread is within the matrix bounds
    for (; row < target_rows && col < target_cols; 
            row += stripe_x, 
            col += stripe_y)
    {
        cu_number_t sum = 0;
#pragma unroll
		for (size_t i = 0; i < m2_rows; i++)
			sum += m1[row * m2_rows + i] * m2[i * m2_cols + col];

		res[row * target_cols + col] = sum;
    }
}

