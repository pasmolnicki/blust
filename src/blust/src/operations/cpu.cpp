#include <blust/backend/cpu_ops.hpp>

#include <sys/time.h>


/*

This file contains implementation of gemm and other 1d operations

Right now the speed of the matrix multiplication is 2.5x slower on my
pc than numpy's.

Further optimization:
- use prefetching
Need to read more about it

- make the register-friendly micro-kernel size 
(there is 16 avx2 registers), but in 8x8 kernel I'm using
17 of them -> reduce the kernel size to 8x6 
(doesn't offer better performance)

- parallelize with openmp

*/

START_BLUST_NAMESPACE

using tensor_t = operations::tensor_t;
using pointer = tensor_t::pointer;
using tensor_rref_t = operations::tensor_rref_t;

typedef union {
    __m128 v;
    float d[4];
} vec4f_t;

typedef union {
    __m256 v;
    float d[8];
} vec8f_t;

/**
 * @brief Make the compiler assume that the `data` is n-byte aligned (n = tensor alignment)
 */
constexpr void assume_aligned(pointer data) 
{
    // if the compiler is gcc or clang, use the builtin function
#if defined(__GNUC__) || defined(__clang__)
    data = (pointer) __builtin_assume_aligned(data, tensor::alignment);
#endif
}

// C'tor
cpu_ops::cpu_ops(int nthreads) : m_ncores(std::max<int>(1, nthreads)) {
    M_realloc_packed(256, 128, 256);
}

cpu_ops::~cpu_ops() 
{
    if (m_aPacked) std::free(m_aPacked);
    if (m_bPacked) std::free(m_bPacked);
}

/**
 * @brief Performs c = a * n + b * m, a,b and c must be 16-byte aligned
 */
void cpu_ops::M_impl_add(
    pointer a_data, pointer b_data, 
    pointer c_data, size_t size, 
    number_t n, number_t m
) noexcept
{
    assume_aligned(a_data);
    assume_aligned(b_data);
    assume_aligned(c_data);

    // with -03 and -mavx2 this is faster
    for (size_t i = 0; i < size; i++) {
        c_data[i] = a_data[i] * n + b_data[i] * m;
    }  
}

/**
 * @brief Performs c = a * b, a, b and c must be n-byte aligned (n = tensor alignment), 
 * optimized to use simd instructions
 */
void cpu_ops::M_impl_hadamard(
    pointer a_data, pointer b_data, 
    pointer c_data, size_t size, 
    number_t /*n*/, number_t /*m*/
) noexcept
{
    assume_aligned(a_data);
    assume_aligned(b_data);
    assume_aligned(c_data);

    // #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        c_data[i] = a_data[i] * b_data[i];
    }
}

constexpr pointer M(pointer m, size_t ldm, size_t y, size_t x) noexcept {
    return m + y * ldm + x;
}

void add_kernel_dot_8x8(
    pointer __restrict a, pointer __restrict b, 
    pointer __restrict c, size_t n, 
    size_t lda, size_t ldb, size_t ldc
) noexcept
{
    assume_aligned(a);
    assume_aligned(b);
    assume_aligned(c);

    vec8f_t 
        va0, va1, va2, va3, va4, va5, va6, va7,
        vb,
        vc0, vc1, vc2, vc3, vc4, vc5, vc6, vc7;
    
    vc0.v = _mm256_setzero_ps(); vc1.v = _mm256_setzero_ps();
    vc2.v = _mm256_setzero_ps(); vc3.v = _mm256_setzero_ps();
    vc4.v = _mm256_setzero_ps(); vc5.v = _mm256_setzero_ps();
    vc6.v = _mm256_setzero_ps(); vc7.v = _mm256_setzero_ps();

    for (size_t i = 0; i < n; i++)
    {
        // Load the first 8 rows of a
        va0.v = _mm256_set1_ps(*M(a, lda, 0, i));
        va1.v = _mm256_set1_ps(*M(a, lda, 1, i));
        va2.v = _mm256_set1_ps(*M(a, lda, 2, i));
        va3.v = _mm256_set1_ps(*M(a, lda, 3, i));
        va4.v = _mm256_set1_ps(*M(a, lda, 4, i));
        va5.v = _mm256_set1_ps(*M(a, lda, 5, i));
        va6.v = _mm256_set1_ps(*M(a, lda, 6, i));
        va7.v = _mm256_set1_ps(*M(a, lda, 7, i));

        // Load column of b (8 elements, at ith row)
        vb.v = _mm256_loadu_ps(M(b, ldb, i, 0));

        // c00 = a0 * b0, c01 = a0 * b1, c02 = a0 * b2, c03 = a0 * b3, ...
        // c10 = a1 * b0, c11 = a1 * b1, c12 = a1 * b2, ...
        vc0.v = _mm256_fmadd_ps(va0.v, vb.v, vc0.v);
        vc1.v = _mm256_fmadd_ps(va1.v, vb.v, vc1.v);
        vc2.v = _mm256_fmadd_ps(va2.v, vb.v, vc2.v);
        vc3.v = _mm256_fmadd_ps(va3.v, vb.v, vc3.v);
        vc4.v = _mm256_fmadd_ps(va4.v, vb.v, vc4.v);
        vc5.v = _mm256_fmadd_ps(va5.v, vb.v, vc5.v);
        vc6.v = _mm256_fmadd_ps(va6.v, vb.v, vc6.v);
        vc7.v = _mm256_fmadd_ps(va7.v, vb.v, vc7.v);
    }

    // Add to previous c vector the result, and store it in c
    _mm256_storeu_ps(M(c, ldc, 0, 0), 
        _mm256_add_ps(_mm256_loadu_ps(M(c, ldc, 0, 0)), vc0.v));

    _mm256_storeu_ps(M(c, ldc, 1, 0), 
        _mm256_add_ps(_mm256_loadu_ps(M(c, ldc, 1, 0)), vc1.v));

    _mm256_storeu_ps(M(c, ldc, 2, 0), 
        _mm256_add_ps(_mm256_loadu_ps(M(c, ldc, 2, 0)), vc2.v));

    _mm256_storeu_ps(M(c, ldc, 3, 0), 
        _mm256_add_ps(_mm256_loadu_ps(M(c, ldc, 3, 0)), vc3.v));

    _mm256_storeu_ps(M(c, ldc, 4, 0), 
        _mm256_add_ps(_mm256_loadu_ps(M(c, ldc, 4, 0)), vc4.v));

    _mm256_storeu_ps(M(c, ldc, 5, 0), 
        _mm256_add_ps(_mm256_loadu_ps(M(c, ldc, 5, 0)), vc5.v));

    _mm256_storeu_ps(M(c, ldc, 6, 0), 
        _mm256_add_ps(_mm256_loadu_ps(M(c, ldc, 6, 0)), vc6.v));

    _mm256_storeu_ps(M(c, ldc, 7, 0), 
        _mm256_add_ps(_mm256_loadu_ps(M(c, ldc, 7, 0)), vc7.v));
}

void scalar_kernel_8x1(
    pointer __restrict a, pointer __restrict b, 
    pointer __restrict c, size_t n, 
    size_t lda, size_t ldb, size_t ldc
)
{
    number_t 
        c00 = 0, c10 = 0, c20 = 0, c30 = 0,
        c40 = 0, c50 = 0, c60 = 0, c70 = 0;
    number_t a0, a1, a2, a3, a4, a5, a6, a7;
    number_t b0;

    for (size_t i = 0; i < n; i++) {
        b0 = *M(b, ldb, i, 0);
        
        // Don't know how to make it vectorized,
        // the 'a' matrix is row-major, so I cannot
        // load the data like a column.
        a0 = *M(a, lda, 0, i);
        a1 = *M(a, lda, 1, i);
        a2 = *M(a, lda, 2, i);
        a3 = *M(a, lda, 3, i);
        a4 = *M(a, lda, 4, i);
        a5 = *M(a, lda, 5, i);
        a6 = *M(a, lda, 6, i);
        a7 = *M(a, lda, 7, i);

        c00 += a0 * b0;
        c10 += a1 * b0;
        c20 += a2 * b0;
        c30 += a3 * b0;
        c40 += a4 * b0;
        c50 += a5 * b0;
        c60 += a6 * b0;
        c70 += a7 * b0;
    }

    *M(c, ldc, 0, 0) += c00;
    *M(c, ldc, 1, 0) += c10;
    *M(c, ldc, 2, 0) += c20;
    *M(c, ldc, 3, 0) += c30;
    *M(c, ldc, 4, 0) += c40;
    *M(c, ldc, 5, 0) += c50;
    *M(c, ldc, 6, 0) += c60;
    *M(c, ldc, 7, 0) += c70;
}

void scalar_kernel_1x8(
    pointer __restrict a, pointer __restrict b, 
    pointer __restrict c, size_t n, 
    size_t lda, size_t ldb, size_t ldc
)
{
    vec8f_t vb, va, vc;
    vc.v = _mm256_setzero_ps();

    for (size_t i = 0; i < n; i++) {
        // b0 = *M(b, ldb, i, 0);
        // b1 = *M(b, ldb, i, 1);
        // ...
        vb.v = _mm256_loadu_ps(M(b, ldb, i, 0));
        
        // a0 = *M(a, lda, 0, i);
        va.v = _mm256_set1_ps(*M(a, lda, 0, i));

        // c00 += a0 * b0;
        // c01 += a0 * b1;
        // ...
        vc.v = _mm256_fmadd_ps(va.v, vb.v, vc.v);
    }

    // *M(c, ldc, 0, 0) += c00;
    // *M(c, ldc, 0, 1) += c01;
    // ...
    _mm256_storeu_ps(M(c, ldc, 0, 0), 
        _mm256_add_ps(_mm256_loadu_ps(M(c, ldc, 0, 0)), vc.v));
}

void packAPanels(number_t* __restrict pack, const number_t* __restrict A,
    size_t m, size_t k, size_t lda) {
    // Assumes m * n is a multiple of 32
    // number_t* pack = utils::aligned_alloc<32, number_t>(m * n);
    const size_t kc_main = (m / 8) * 8;
    size_t i, j;

    for (i = 0; i < m; i++) {
        for (j = 0; j < kc_main; j += 8) {
            // Load 8 elements from A
            vec8f_t va;
            va.v = _mm256_loadu_ps(&A[i * lda + j]);

            // Store them in packed format
            _mm256_storeu_ps(pack, va.v);
            pack += 8;
        }

        // Handle the remaining columns
        for (; j < k; j++) {
            *pack++ = A[i * lda + j];
        }
    }
}

void packRowMajor(number_t* pack, const number_t* A, size_t m, size_t k, size_t lda) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < k; j++) {
            // pack[i * k + j] = A[i * lda + j];
            *pack++ = A[i * lda + j];
        }
    }
}

// Calculates C = A * B, where each matrix is row-major and 
// dimensions are: A: m x k, B: k x n, C: m x n, 
// in standard case lda = k, ldb = n, ldc = n
// 
// kernel calculates mini matrix multiplication (using kernelM x kernelN size)
template <size_t kernelM, size_t kernelN>
void cpu_ops::M_inner_kernel(
    size_t m, size_t k, size_t n, pointer __restrict A, 
    pointer __restrict B, pointer __restrict C, 
    size_t lda, size_t ldb, size_t ldc,
    size_t MC, size_t KC, size_t NC,
    cpu_ops::func_kernel_dot_t kernel,
    func_scalar_kernel_t kernel_1xN,
    func_scalar_kernel_t kernel_Nx1
) noexcept
{
    constexpr auto MR = kernelM, NR = kernelN;

    for (size_t N0 = 0; N0 < n; N0 += NC) {
        const size_t nc = std::min(NC, n - N0);
        for (size_t K0 = 0; K0 < k; K0 += KC) {
            const size_t kc = std::min(KC, k - K0);

            // Pack B(K0:K0+kc, N0:N0+nc) into bPacked
            packRowMajor(m_bPacked, &B[K0 * ldb + N0], kc, nc, ldb);

            for (size_t M0 = 0; M0 < m; M0 += MC) {
                size_t mc = std::min(MC, m - M0);

                // Pack A(M0:M0+mc, K0:K0+kc) to Packed
                packRowMajor(m_aPacked, &A[M0 * lda + K0], mc, kc, lda);

                // if (M0 + MC < m) {
                //     _mm_prefetch((const char*)&A[(M0 + MC) * lda + K0], _MM_HINT_T0);
                // }

                // If mc is not a multiple of MR, we must 
                // handle the rest of the rows, same for kc
                // so this simply get the 'multiple of MR' part from mc
                const size_t mc_main = (mc / MR) * MR;
                const size_t nc_main = (nc / NR) * NR;

                 // Multiply A(M0:M0+mc, N0:N0+nc) * B(N0:N0+nc, K0:K0+kc) = C(M0:M0+mc, K0:K0+kc)
                for (size_t ir = 0; ir < mc_main; ir += MR) {
                    for (size_t jr = 0; jr < nc_main; jr += NR) {
                        
                        // submatrix of a(M0:M0+mc, K0:K0+kc), lda_sub = kc
                        pointer a_sub = &m_aPacked[ir * kc + 0];
                        // submatrix of b(K0:K0+kc, N0:N0+nc), ldb_sub = nc
                        pointer b_sub = &m_bPacked[0 * nc + jr];
                        // c matrix
                        pointer c_sub = &C[jr + N0 + (M0 + ir) * ldc];

                        // n (inner-product length) is nc
                        kernel(
                            a_sub, b_sub, c_sub,
                            kc, kc, nc, ldc
                        );
                    }

                    for (size_t j = nc_main; j < nc; j++) {
                        kernel_Nx1(
                            &m_aPacked[ir * kc + 0],
                            &m_bPacked[0 * nc + j],
                            &C[j + N0 + (M0 + ir) * ldc],
                            kc, kc, nc, ldc
                        );
                    }
                }

                // Bottom-edge
                for (size_t i = mc_main; i < mc; i++) {
                    for (size_t jr = 0; jr < nc_main; jr+=NR) {
                        kernel_1xN(
                            &m_aPacked[i * kc + 0],
                            &m_bPacked[0 * nc + jr],
                            &C[jr + N0 + (M0 + i) * ldc],
                            kc, kc, nc, ldc
                        );
                    }

                    // Corner
                    for (size_t j = nc_main; j < nc; j++) {
                        number_t sum = 0;
                        for (size_t p = 0; p < kc; p++) {
                            sum += m_aPacked[i * kc + p] * m_bPacked[p * nc + j];
                        }
                        C[(M0 + i) * ldc + (j + K0)] += sum;
                    }
                }
            }
        }
    }
}

// I know this is pointless, since if the cpu doesn't support avx2,
// it won't compile (i think), but I want to keep the code clean
void cpu_ops::M_impl_matumul(
    pointer __restrict a, size_t lda, 
    pointer __restrict b, size_t ldb,
    pointer __restrict c, size_t ldc,
    size_t m, size_t k, size_t n,
    size_t MC, size_t KC, size_t NC
) noexcept
{
    M_realloc_packed(MC, KC, NC);

    M_inner_kernel<8, 8>(
        m, k, n, 
        a, b, c, 
        lda, ldb, ldc,
        MC, KC, NC,
        add_kernel_dot_8x8, 
        scalar_kernel_1x8,
        scalar_kernel_8x1
    );
}

void cpu_ops::M_realloc_packed(size_t MC, size_t KC, size_t NC) noexcept
{
    if (M_MC == MC && M_KC == KC && M_NC == NC)
        return;

    // A different size, reallocate
    if (M_MC != MC || M_KC != KC) {
        if (m_aPacked) std::free(m_aPacked);
        m_aPacked = utils::aligned_alloc<32, number_t>(MC * KC);
    }

    if (M_NC != NC || M_KC != KC) {
        if (m_bPacked) std::free(m_bPacked);
        m_bPacked = utils::aligned_alloc<32, number_t>(KC * NC);
    }

    M_MC = MC;
    M_KC = KC;
    M_NC = NC;
}


// Get the result tensor, based on the a's and b's operation flags.
// Asserts same size of a and b, then tries to borrow a's or b's buffers
inline tensor_t cpu_ops::M_get_res_tensor(tensor_t& a, tensor_t& b)
{
    // Assert same size
    M_assert_tensor_same_size(a, b);

    // Try to borrow buffers, to avoid redundand memory allocation
    tensor_t res = ops_tensor::try_borrow(a, b);

    // if in chained operation, will use that fact 
    // in next 'operation' this tensor is used
    res.set_in_operation(true); 
    return res;
}

/**
 * @brief Perform the vector-like operation, with the given function
 * (either `M_impl_add` or `M_impl_hadamard`), may launch threads 
 * if the result's size is big enough
 * @return the result tensor
 */
inline tensor_t cpu_ops::M_perform_vector_like(
    tensor_t& a, tensor_t& b, 
    number_t n, number_t m, 
    func_vector_t func,
    bool allocate
)
{
    BLUST_ASSERT(a.dim()==b.dim());

    tensor_t res;
    if (allocate) {
        res = M_get_res_tensor(a, b);
    } else {
        // Share buffer with a
        res.M_borrow(a);
    }

    const auto size = res.size();
    if (M_should_lanuch_threads(size)) 
    {
        // dispatch threads to do the work in parallel
        int offset_size = size / m_ncores;
        int offset = 0;
        auto a_data = a.data();
        auto b_data = b.data();
        auto res_data = res.data();

        for (int i = 0; i < m_ncores; i++, offset += offset_size) {
            size_t patch_size = i == m_ncores - 1 ? size - offset : offset_size;
            m_threads.push_back(
                std::thread(
                    func, a_data + offset, b_data + offset, 
                    res_data + offset, patch_size, n, m
                ));
        }

        M_join_threads();
    }
    else
        func(a.data(), b.data(), res.data(), size, n, m);

    return res;
}

/**
 * @brief Add two tensors and return the result
 */
tensor_rref_t cpu_ops::add(tensor_t a, tensor_t b, bool allocate) 
{
    return M_perform_vector_like(a, b, 1.0, 1.0, M_impl_add, allocate);
}

/**
 * @brief Perform substaction (a - b) and return the result
 */
tensor_rref_t cpu_ops::sub(tensor_t a, tensor_t b, bool allocate) 
{
    return M_perform_vector_like(a, b, 1.0, -1.0, M_impl_add, allocate);
}

/**
 * @brief Caluculate Ri = Ai * b (see hadamard for element-wise multiplication)
 */
tensor_rref_t cpu_ops::mul(tensor_t a, number_t b, bool allocate) 
{
    return M_perform_vector_like(a, a, b, 0.0, M_impl_add, allocate); // c = a * b + a * 0
}

/**
 * @brief Calculate Ri = Ai / b
 */
tensor_rref_t cpu_ops::div(tensor_t a, number_t b, bool allocate) 
{
    return M_perform_vector_like(a, a, 1 / b, 0.0, M_impl_add, allocate);
}

/**
 * @brief Get the hadamard product: Ci = Ai * Bi
 */
tensor_rref_t cpu_ops::hadamard(tensor_t a, tensor_t b, bool allocate)
{
    return M_perform_vector_like(a, b, 0, 0, M_impl_hadamard, allocate);
}

/**
 * @brief Perform matrix multiplication, and return the result
 * @param a the first matrix, with dimensions m x n and in a column-major order
 * @param b the second matrix, with dimensions n x k and in a column-major order
 * @return the result matrix, with dimensions m x k and in a column-major order
 */
tensor_rref_t cpu_ops::mat_mul(tensor_t a, tensor_t b, size_t MC, size_t KC, size_t NC)
{
    M_assert_tensor_dim_mat_mul(a, b);

    const size_t m = a.dim()[0];
    const size_t k = a.dim()[1];
    const size_t n = b.dim()[1];

    auto res = ops_tensor({m, n});

    M_impl_matumul(
        a.data(), k,
        b.data(), n,
        res.data(), n,
        m, k, n,
        M_MC, M_KC, M_NC
    );

    return std::move(res);
}

// https://stackoverflow.com/questions/16737298/what-is-the-fastest-way-to-transpose-a-matrix-in-c
inline void transpose4x4_SSE(float *A, float *B, const int lda, const int ldb) {
    __m128 row1 = _mm_load_ps(&A[0*lda]);
    __m128 row2 = _mm_load_ps(&A[1*lda]);
    __m128 row3 = _mm_load_ps(&A[2*lda]);
    __m128 row4 = _mm_load_ps(&A[3*lda]);
    _MM_TRANSPOSE4_PS(row1, row2, row3, row4);
    _mm_store_ps(&B[0*ldb], row1);
    _mm_store_ps(&B[1*ldb], row2);
    _mm_store_ps(&B[2*ldb], row3);
    _mm_store_ps(&B[3*ldb], row4);
}

inline void transpose_block_SSE4x4(float *A, float *B, const int n, const int m, const int lda, const int ldb ,const int block_size) {
    #pragma omp parallel for
    for(int i=0; i<n; i+=block_size) {
        for(int j=0; j<m; j+=block_size) {
            int max_i2 = i+block_size < n ? i + block_size : n;
            int max_j2 = j+block_size < m ? j + block_size : m;
            for(int i2=i; i2<max_i2; i2+=4) {
                for(int j2=j; j2<max_j2; j2+=4) {
                    transpose4x4_SSE(&A[i2*lda +j2], &B[j2*ldb + i2], lda, ldb);
                }
            }
        }
    }
}

tensor_rref_t cpu_ops::transpose(tensor_t a)
{
    const size_t n_rows = a.dim()[0];
    const size_t n_cols = a.dim()[1];

    auto res = ops_tensor({n_cols, n_rows});

    transpose_block_SSE4x4(
        a.data(), res.data(),
        n_rows, n_cols,
        n_cols, n_rows,
        64
    );

    return std::move(res);
}

END_BLUST_NAMESPACE