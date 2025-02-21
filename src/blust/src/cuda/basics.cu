#include <iostream>
#include <math.h>

typedef float number_t;
typedef number_t* pointer_t;


// VECTOR OPERATIONS

// Should be launched with maximum of 32 * 256 threads (obviously based on the data size)
// with 1d parameters (Nbocks, Block size)
__global__ void vector_add(pointer_t m1, pointer_t m2, size_t N)
{
    auto i      = threadIdx.x + blockDim.x * blockIdx.x;
    auto stripe = blockDim.x * gridDim.x;
    for (; i < N; i+=stripe)
        m1[i] += m2[i];
}

// Should be launched with 1d parameters
__global__ void vector_sub(pointer_t m1, pointer_t m2, size_t N)
{
    auto i      = threadIdx.x + blockDim.x * blockIdx.x;
    auto stripe = blockDim.x * gridDim.x;
    for (; i < N; i+=stripe)
        m1[i] -= m2[i];
}

// Multiplies A * B, such that Cij = Aij * Bij
__global__ void vector_mul_hadamard(pointer_t m1, pointer_t m2, size_t N)
{
    auto i      = threadIdx.x + blockDim.x * blockIdx.x;
    auto stripe = blockDim.x * gridDim.x;
    for (; i < N; i+=stripe)
        m1[i] *= m2[i];
}


// Transpose the matrix `m`, store the result in `res`
__global__ void mat_transpose(pointer_t m, pointer_t res, size_t rows, size_t cols)
{
    const auto stripe = blockDim.x * gridDim.x;
    const auto N      = rows * cols;
    for (size_t n = blockDim.x * blockIdx.x + threadIdx.x; n < N; n += stripe)
    {
        int i = n / rows;
        int j = n % rows;
        res[n] = m[cols*j + i];
    }
}

// Multiply two matrices, and store the result in `res`
__global__ void mat_mul(pointer_t m1, pointer_t m2, pointer_t res, size_t m1_rows, size_t m2_cols, size_t m2_rows)
{

}

__global__
void add(int n, float *x, float *y)
{
    for (int i = 0; i < n; ++i)
        y[i] += x[i];
}

int main()
{
    int N = 1 << 20;

    float *x, *y;

    // x = new float[N];
    // y = new float[N];
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    for (int i = 0; i < N; ++i)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    add<<<1, 1>>>(N, x, y);

    float maxErr = 0.0f;
    for (int i = 0; i < N; ++i)
        maxErr = fmax(maxErr, fabs(y[i] - 3.0f));
    std::cout << "Max err: " << maxErr << '\n';

    cudaFree(x);
    cudaFree(y);

    return 0;
}
