typedef float number_t;

__kernel void vector_add(
    __global number_t* A,
    __global number_t* B,
    __global number_t* C,
    unsigned int N,
    number_t alpha,
    number_t beta
)
{
    unsigned int i = get_global_id(0);
    if (i < N) {
        C[i] = A[i] * alpha + B[i] * beta;
    }
}


__kernel void vector_hadamard(
    __global number_t* A,
    __global number_t* B,
    __global number_t* C,
    unsigned int N
)
{
    unsigned int i = get_global_id(0);
    if (i < N) {
        C[i] = A[i] * B[i];
    }
}


#define TILE_SIZE 16

__kernel void matrix_mul(
    __global const number_t* A,
    __global const number_t* B,
    __global number_t* C,
    const unsigned int M,
    const unsigned int K,
    const unsigned int N
)
{
    unsigned int local_x = get_local_id(0);
    unsigned int local_y = get_local_id(1);

    unsigned int group_x = get_group_id(0);
    unsigned int group_y = get_group_id(1);

    unsigned int row = group_y * TILE_SIZE + local_y;
    unsigned int col = group_x * TILE_SIZE + local_x;

    unsigned int lda = K;
    unsigned int ldb = N;
    unsigned int ldc = N;

    __local number_t A_tile[TILE_SIZE][TILE_SIZE];
    __local number_t B_tile[TILE_SIZE][TILE_SIZE];

    number_t sum = 0.0f;

    for (int i = 0; i < (K + TILE_SIZE - 1) / TILE_SIZE; i++) {
        if (row < M && (i * TILE_SIZE + local_x) < K)
            A_tile[local_y][local_x] = A[row * lda + (i * TILE_SIZE + local_x)];
        else
            A_tile[local_y][local_x] = 0.0f;

        if (col < N && (i * TILE_SIZE + local_y) < K)
            B_tile[local_y][local_x] = B[(i * TILE_SIZE + local_y) * ldb + col];
        else
            B_tile[local_y][local_x] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j = 0; j < TILE_SIZE; j++) {
            sum += A_tile[local_y][j] * B_tile[j][local_x];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < M &&  col < N) {
        C[row * ldc + col] = sum;
    }
}