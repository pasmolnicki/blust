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
    const int local_x = get_local_id(1);
    const int local_y = get_local_id(0);

    const int global_x = get_group_id(0) * TILE_SIZE + local_x;
    const int global_y = get_group_id(1) * TILE_SIZE + local_y;

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
#define TILE_SIZE_REG 32

// This is the number of column threads
#define RTS 4

// Work-per-thread (one dimension) (TILE_SIZE_REG/RTS)
#define WPT 8


// Block only by the column, each thread will compute one 
// the row of C  (total `WPT` elements) 
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

    const unsigned int local_y = get_local_id(0); // max TILE_SIZE_REG
    const unsigned int local_x = get_local_id(1); // 0..RTS-1

    // Linear thread ID for loading (0..31)
    const int tid = local_y * RTS + local_x; 

    // Because each work-group has assigned TILE_SIZE_REG of work (with only 8x8 threads)
    // I map the group_x and group_y (starting global C index of this work-group) this way
    // it points to the top-left corner of the tile this work-group is working on.
    // Starting global index of C for each thread (a block)
    const unsigned int global_y = get_group_id(0) * TILE_SIZE_REG + local_y;
    const unsigned int global_x = get_group_id(1) * TILE_SIZE_REG + local_x;
    const int n_tiles = (K + TILE_SIZE_REG - 1) / TILE_SIZE_REG;

    number_t acc[WPT];
    for (int i = 0; i < WPT; i++) {
        acc[i] = 0;
    }


    for (int t = 0; t < n_tiles; t++) {
        const int tile_offset = t * TILE_SIZE_REG;
        const int num_elements_to_load = (TILE_SIZE_REG * TILE_SIZE_REG) / (TILE_SIZE_REG * RTS);

        for (int i = 0; i < num_elements_to_load; i++) {
            // 1. Calculate which pixel (t_row, t_col) of the tile this thread handles
            // We map threads linearly across the tile for better coalescing
            int linear_idx = tid + i * (TILE_SIZE_REG * RTS);
            
            int t_row = linear_idx / TILE_SIZE_REG;
            int t_col = linear_idx % TILE_SIZE_REG;

            // 2. Load A (Coalesced if possible)
            // Global A Row: GroupY_Base + t_row
            // Global A Col: tile_offset + t_col
            int global_A_row = get_group_id(0) * TILE_SIZE_REG + t_row;
            int global_A_col = tile_offset + t_col;

            if (global_A_row < M && global_A_col < K)
                A_tile[t_row][t_col] = A[global_A_row * K + global_A_col];
            else
                A_tile[t_row][t_col] = 0.0f;

            // 3. Load B (Coalesced)
            // Global B Row: tile_offset + t_row
            // Global B Col: GroupX_Base + t_col
            int global_B_row = tile_offset + t_row;
            int global_B_col = get_group_id(1) * TILE_SIZE_REG + t_col;

            if (global_B_row < K && global_B_col < N)
                B_tile[t_row][t_col] = B[global_B_row * N + global_B_col];
            else
                B_tile[t_row][t_col] = 0.0f;
        }

        sync_threads();

        // Calculate sub-matrix result, the A_tiled x B_tiled
        for (int k = 0; k < TILE_SIZE; k++) {
            for (int w = 0; w < WPT; w++) {
                acc[w] += A_tile[local_y][k] * B_tile[k][local_x + w * RTS];
            }
        }

        sync_threads();
    }

    if (global_y < M) {
        for (int w = 0; w < WPT; w++) {
            const int x = global_x + w * RTS;
            if (x < N) {
                C[global_y * N + x] = acc[w];
            }
        }
    }
}