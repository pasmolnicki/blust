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

#define sync_threads() barrier(CLK_LOCAL_MEM_FENCE)

#define TILE_SIZE 16

// Should be lanuched with TILE_SIZE x TILE_SIZE threads
__kernel void mat_mul_tiled(
    __global const number_t* A,
    __global const number_t* B,
    __global number_t* C,
    const unsigned int M,
    const unsigned int K,
    const unsigned int N
)
{
    const int global_x = get_global_id(0);
    const int global_y = get_global_id(1);
    const int local_x = get_local_id(0);
    const int local_y = get_local_id(1);

    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    __local number_t A_tile[TILE_SIZE][TILE_SIZE];
    __local number_t B_tile[TILE_SIZE][TILE_SIZE];

    number_t sum = 0.0f;
    const int n_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < n_tiles; t++) {
        // Copy into A_tiled and B_tiled
        const int tile_offset = t * TILE_SIZE;
        
        // Each thread copies exactly one value from the original matrix
        // into the shared memory. It can does that because the size of 
        // work-group is exactly the size of the tiled matrices
        // A: M x K
        if (global_y < M && (tile_offset + local_x) < K) {
            A_tile[local_y][local_x] = A[global_y * lda + (tile_offset + local_x)];
        } else {
            A_tile[local_y][local_x] = 0;
        }

        // B: K x N
        if ((tile_offset + local_y) < K && global_x < N) {
            B_tile[local_y][local_x] = B[(tile_offset + local_y) * ldb + global_x];
        } else {
            B_tile[local_y][local_x] = 0;
        }

        sync_threads();

        // Calculate sub-matrix result, the A_tiled x B_tiled
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += A_tile[local_y][k] * B_tile[k][local_x];
        }

        sync_threads();
    }

    if (global_y < M && global_x < N) {
        C[global_y * ldc + global_x] = sum;
    }
}


// This is the work group tile size, the theads are organized
// in (TILE_SIZE_REG x TILE_SIZE_REG) blocks, and each thread computes
// a BLOCK_SIZE x BLOCK_SIZE sub-block of the C matrix.
#define TILE_SIZE_REG 64


// This is the block size each thread computes
#define BLOCK_SIZE 4

// Will be lanuched with a 2D grid of work-groups, each of size
// 8x8 (64 threads), so that each work-group computes a 
// TILE_SIZE_REG x TILE_SIZE_REG (32x32) sub-matrix of C. With each thread
// calculating a BLOCK_SIZE x BLOCK_SIZE (4x4) sub-block of C.
// That's why this kernel must be lanuched with a (TILE_SIZE_REG/BLOCK_SIZE)^2
__kernel void mat_mul_register_blocking(
    __global const number_t* A,
    __global const number_t* B,
    __global number_t* C,
    const unsigned int M,
    const unsigned int K,
    const unsigned int N
)
{
    __local number_t A_tile[TILE_SIZE_REG][TILE_SIZE_REG];
    __local number_t B_tile[TILE_SIZE_REG][TILE_SIZE_REG];

    const unsigned int local_x = get_local_id(0); // 0..7
    const unsigned int local_y = get_local_id(1); // 0..7

    // Because each work-group has assigned TILE_SIZE_REG of work (with only 8x8 threads)
    // I map the group_x and group_y (starting global C index of this work-group) this way
    // it points to the top-left corner of the tile this work-group is working on.
    const unsigned int group_y = get_group_id(0) * TILE_SIZE_REG;
    const unsigned int group_x = get_group_id(1) * TILE_SIZE_REG;

    // Starting global index of C for each thread (a block)
    const unsigned int global_block_y = group_y + local_y * BLOCK_SIZE;
    const unsigned int global_block_x = group_x + local_x * BLOCK_SIZE;

    // Accumulator for the C value, initialized to zero
    number_t results[BLOCK_SIZE][BLOCK_SIZE] = {0};

    const int num_tiles = (K + TILE_SIZE_REG - 1) / TILE_SIZE_REG;
    for (int t = 0; t < num_tiles; t++) {
        const int tile_offset = t * TILE_SIZE_REG;

        // Load A and B tiles into local memory,
        // each thread must load multiple elements
        // i.e. with TILE_SIZE_REG=32 (32x32) and 8x8 work-group,
        // each thread loads 4 elements from A and 4 from B

        // total elements each thread must copy
        const int total_per_thread = (TILE_SIZE_REG * TILE_SIZE_REG) / (get_local_size(0) * get_local_size(1));
        
        // This is the linear work-group thread id
        const int thread_local_id = local_y * get_local_size(0) + local_x;

        // Instead of doing something like this:
        // x_elems = TILE_SIZE_REG / get_local_size(0)
        // y_elems = TILE_SIZE_REG / get_local_size(0)
        // for y in 0..y_elems:
        //      for x in 0..x_elems:
        //          A_tiled[y + block_offset_y][x + block_offset_x] = A[global_y][global_x]
        // etc.
        // That would be making copy with stride of BLOCK_SIZE, which would dramatically
        // drag the performance down. 
        // C . . . C . . . C . . . (...)
        // C . . . C . . . C . . . (...)
        // ...
        for (int y = local_y; y < TILE_SIZE_REG; y += get_local_size(1)) {
            for (int x = local_x; x < TILE_SIZE_REG; x += get_local_size(0)) {
                int global_x = group_x + x;
                int global_y = group_y + y;
                int a_x = tile_offset + x;
                int b_y = tile_offset + y;

                if (global_y < M && a_x < K) {
                    A_tile[y][x] = A[global_y * K + a_x];
                } else {
                    A_tile[y][x] = 0;
                }

                if (b_y < K && global_x < N) {
                    B_tile[y][x] = B[b_y * N + global_x];
                } else {
                    B_tile[y][x] = 0;
                }
            }
        }

        // Instead I opt for linear copy:
        // C C C C C C C C C C C C (...)
        // . . . . . . . . . . . . (...)
        // ...
        // Perform a work-group copy (1x64) of the A and B into shared memory
        // (Iterating over the larger TILE_SIZE_REG x TILE_SIZE_REG tile)
        // for (int i = 0; i < total_per_thread; i++) {
        //     int linear_idx = thread_local_id + i * get_local_size(0) * get_local_size(1);

        //     int tile_x = linear_idx % TILE_SIZE_REG;
        //     int tile_y = linear_idx / TILE_SIZE_REG;

        //     // Tile starting index + tile index = global index
        //     int global_x = group_x + tile_x;
        //     int global_y = group_y + tile_y;

        //     // A: M x K, k-th index is the tile_offset + tile_y
        //     if (global_y < M && (tile_offset + tile_x) < K) {
        //         A_tile[tile_y][tile_x] = A[global_y * K + (tile_offset + tile_x)];
        //     } else {
        //         A_tile[tile_y][tile_x] = 0;
        //     }

        //     // B: K x N, k-th index is the same as A's
        //     if ((tile_offset + tile_y) < K && global_x < N) {
        //         B_tile[tile_y][tile_x] = B[(tile_offset + tile_y) * N + global_x];
        //     } else {
        //         B_tile[tile_y][tile_x] = 0;
        //     }
        // }

        sync_threads();

        // Now perform the calculations, each thread will calculate 
        // a sub-matrix multiplication of size:
        // A-sub: BLOCK_SIZE x TILE_SIZE_REG
        // B-sub: TILE_SIZE_REG x BLOCK_SIZE
        // result: C BLOCK_SIZE x BLOCK_SIZE
        for (int k = 0; k < TILE_SIZE_REG; k++) {
            // Copy into BLOCK_SIZE float registers
            number_t a_reg[BLOCK_SIZE];
            number_t b_reg[BLOCK_SIZE];

            // Copy (k=0) a column of BLOCK_SIZE 
            // elements from A_tiled into registers
            // C . . . 
            // C . . . 
            // C . . .
            // C . . .
            #pragma unroll
            for (int i = 0; i < BLOCK_SIZE; i++) {
                a_reg[i] = A_tile[i + local_y * BLOCK_SIZE][k];
            }

            // Copy (k=0) a row of BLOCK_SIZE
            // elements from B_tiled into registers
            // C C C C
            // . . . .
            // . . . .
            // . . . .
            #pragma unroll
            for (int i = 0; i < BLOCK_SIZE; i++) {
                b_reg[i] = B_tile[k][i + local_x * BLOCK_SIZE];
            }

            // Now do the damn matrix multplication
            #pragma unroll
            for (int m = 0; m < BLOCK_SIZE; m++) {
                #pragma unroll
                for (int n = 0; n < BLOCK_SIZE; n++) {
                    results[m][n] += a_reg[m] * b_reg[n];
                }
            }
        }

        sync_threads();
    }


    // Copy the results block to the global C matrix
    #pragma unroll
    for (int m = 0; m < BLOCK_SIZE; m++) {
        #pragma unroll
        for (int n = 0; n < BLOCK_SIZE; n++) {
            int x = global_block_x + n;
            int y = global_block_y + m;

            if (y < M && x < N) {
                C[y * N + x] = results[m][n];
            }
        }
    }
}