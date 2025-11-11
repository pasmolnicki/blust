#include <blust/backend/cpu_ops.hpp>

#include <sys/time.h>

START_BLUST_NAMESPACE

typedef operations::tensor_t tensor_t;
typedef tensor_t::pointer pointer;
typedef operations::tensor_rref_t tensor_rref_t;

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
cpu_ops::cpu_ops(int nthreads) : m_ncores(std::max<int>(1, nthreads)) 
{
    aPacked = utils::aligned_alloc<32, number_t>(MC * NC);
    bPacked = utils::aligned_alloc<32, number_t>(NC * KC);
}

cpu_ops::~cpu_ops() 
{
    std::free(aPacked);
    std::free(bPacked);
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
    #pragma omp parallel for
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

    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        c_data[i] = a_data[i] * b_data[i];
    }
}

constexpr pointer M(pointer m, size_t ldm, size_t y, size_t x) noexcept {
    return m + y * ldm + x;
}


void cpu_ops::M_add_kernel_dot_8x8(
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
        vb.v = _mm256_load_ps(M(b, ldb, i, 0));

        // c00 = a0 * b0, c01 = a0 * b1, c02 = a0 * b2, c03 = a0 * b3
        // vc0.v = _mm256_add_ps(vc0.v, _mm256_mul_ps(va0.v, vb.v));
        // vc1.v = _mm256_add_ps(vc1.v, _mm256_mul_ps(va1.v, vb.v));
        // vc2.v = _mm256_add_ps(vc2.v, _mm256_mul_ps(va2.v, vb.v));
        // vc3.v = _mm256_add_ps(vc3.v, _mm256_mul_ps(va3.v, vb.v));
        // vc4.v = _mm256_add_ps(vc4.v, _mm256_mul_ps(va4.v, vb.v));
        // vc5.v = _mm256_add_ps(vc5.v, _mm256_mul_ps(va5.v, vb.v));
        // vc6.v = _mm256_add_ps(vc6.v, _mm256_mul_ps(va6.v, vb.v));
        // vc7.v = _mm256_add_ps(vc7.v, _mm256_mul_ps(va7.v, vb.v));
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
    _mm256_store_ps(M(c, ldc, 0, 0), 
        _mm256_add_ps(_mm256_load_ps(M(c, ldc, 0, 0)), vc0.v));

    _mm256_store_ps(M(c, ldc, 1, 0), 
        _mm256_add_ps(_mm256_load_ps(M(c, ldc, 1, 0)), vc1.v));

    _mm256_store_ps(M(c, ldc, 2, 0), 
        _mm256_add_ps(_mm256_load_ps(M(c, ldc, 2, 0)), vc2.v));

    _mm256_store_ps(M(c, ldc, 3, 0), 
        _mm256_add_ps(_mm256_load_ps(M(c, ldc, 3, 0)), vc3.v));

    _mm256_store_ps(M(c, ldc, 4, 0), 
        _mm256_add_ps(_mm256_load_ps(M(c, ldc, 4, 0)), vc4.v));

    _mm256_store_ps(M(c, ldc, 5, 0), 
        _mm256_add_ps(_mm256_load_ps(M(c, ldc, 5, 0)), vc5.v));

    _mm256_store_ps(M(c, ldc, 6, 0), 
        _mm256_add_ps(_mm256_load_ps(M(c, ldc, 6, 0)), vc6.v));

    _mm256_store_ps(M(c, ldc, 7, 0), 
        _mm256_add_ps(_mm256_load_ps(M(c, ldc, 7, 0)), vc7.v));
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
    number_t c00 = 0, c01 = 0, c02 = 0, c03 = 0,
            c04 = 0, c05 = 0, c06 = 0, c07 = 0;
    number_t b0, b1, b2, b3, b4, b5, b6, b7;
    number_t a0;

    for (size_t i = 0; i < n; i++) {
        b0 = *M(b, ldb, i, 0);
        b1 = *M(b, ldb, i, 1);
        b2 = *M(b, ldb, i, 2);
        b3 = *M(b, ldb, i, 3);
        b4 = *M(b, ldb, i, 4);
        b5 = *M(b, ldb, i, 5);
        b6 = *M(b, ldb, i, 6);
        b7 = *M(b, ldb, i, 7);
        
        a0 = *M(a, lda, 0, i);

        c00 += a0 * b0;
        c01 += a0 * b1;
        c02 += a0 * b2;
        c03 += a0 * b3;
        c04 += a0 * b4;
        c05 += a0 * b5;
        c06 += a0 * b6;
        c07 += a0 * b7;
    }

    *M(c, ldc, 0, 0) += c00;
    *M(c, ldc, 0, 1) += c01;
    *M(c, ldc, 0, 2) += c02;
    *M(c, ldc, 0, 3) += c03;
    *M(c, ldc, 0, 4) += c04;
    *M(c, ldc, 0, 5) += c05;
    *M(c, ldc, 0, 6) += c06;
    *M(c, ldc, 0, 7) += c07;
}

void scalar_kernel_vec1x8(
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
        vb.v = _mm256_load_ps(M(b, ldb, i, 0));
        
        // a0 = *M(a, lda, 0, i);
        va.v = _mm256_set1_ps(*M(a, lda, 0, i));

        // c00 += a0 * b0;
        // c01 += a0 * b1;
        // ...
        // vc.v = _mm256_add_ps(vc.v, _mm256_mul_ps(va.v, vb.v));
        vc.v = _mm256_fmadd_ps(va.v, vb.v, vc.v);
    }

    // *M(c, ldc, 0, 0) += c00;
    // *M(c, ldc, 0, 1) += c01;
    // ...
    auto old_c = _mm256_load_ps(M(c, ldc, 0, 0));
    vc.v = _mm256_add_ps(old_c, vc.v);
    _mm256_store_ps(M(c, ldc, 0, 0), vc.v);
}

// Vectorized matrix multiplication for
// a[0:4, 0:n] * b[0:n, 0:4] = c[0:4, 0:4] matrices
// uses see3 instructions
void cpu_ops::M_add_kernel_dot_4x4(
    pointer __restrict a, pointer __restrict b, 
    pointer __restrict c, size_t n, 
    size_t lda, size_t ldb, size_t ldc
) noexcept
{
    assume_aligned(a);
    assume_aligned(b);
    assume_aligned(c);

    vec4f_t 
        va0, va1, va2, va3, // 4 rows with 4 elements of equal value
        vb, // 1 rows with 4 column elements
        vc0, vc1, vc2, vc3;  // 4 rows of c (with 4 elements)

    vc0.v = _mm_setzero_ps();
    vc1.v = _mm_setzero_ps();
    vc2.v = _mm_setzero_ps();
    vc3.v = _mm_setzero_ps();

    for(size_t i = 0; i < n; i++)
    {
        // Load the row of a
        va0.v = _mm_set1_ps(*M(a, lda, 0, i));
        va1.v = _mm_set1_ps(*M(a, lda, 1, i));
        va2.v = _mm_set1_ps(*M(a, lda, 2, i));
        va3.v = _mm_set1_ps(*M(a, lda, 3, i));
        
        // Load the column of b
        vb.v = _mm_load_ps(M(b, ldb, i, 0));
        // b0, b1, b2, b3

        
        // c00 = a0 * b0, c01 = a0 * b1, c02 = a0 * b2, c03 = a0 * b3
        
        vc0.v = _mm_add_ps(vc0.v, _mm_mul_ps(va0.v, vb.v));
        vc1.v = _mm_add_ps(vc1.v, _mm_mul_ps(va1.v, vb.v));
        vc2.v = _mm_add_ps(vc2.v, _mm_mul_ps(va2.v, vb.v));
        vc3.v = _mm_add_ps(vc3.v, _mm_mul_ps(va3.v, vb.v));
    }

    // Store the result (4 rows of c)
    auto vc_old = _mm_load_ps(M(c, ldc, 0, 0));
    vc0.v = _mm_add_ps(vc_old, vc0.v);
    _mm_store_ps(M(c, ldc, 0, 0), vc0.v);

    vc_old = _mm_load_ps(M(c, ldc, 1, 0));
    vc1.v = _mm_add_ps(vc_old, vc1.v);
    _mm_store_ps(M(c, ldc, 1, 0), vc1.v);

    vc_old = _mm_load_ps(M(c, ldc, 2, 0));
    vc2.v = _mm_add_ps(vc_old, vc2.v);
    _mm_store_ps(M(c, ldc, 2, 0), vc2.v);

    vc_old = _mm_load_ps(M(c, ldc, 3, 0));
    vc3.v = _mm_add_ps(vc_old, vc3.v);
    _mm_store_ps(M(c, ldc, 3, 0), vc3.v);
}

void scalar_kernel_4x1(
    pointer __restrict a, pointer __restrict b, 
    pointer __restrict c, size_t n, 
    size_t lda, size_t ldb, size_t ldc
)
{
    number_t c00 = 0, c10 = 0, c20 = 0, c30 = 0;
    number_t a0, a1, a2, a3;
    number_t b0;

    for (size_t i = 0; i < n; i++) {
        b0 = *M(b, ldb, i, 0);
        
        a0 = *M(a, lda, 0, i);
        a1 = *M(a, lda, 1, i);
        a2 = *M(a, lda, 2, i);
        a3 = *M(a, lda, 3, i);

        c00 += a0 * b0;
        c10 += a1 * b0;
        c20 += a2 * b0;
        c30 += a3 * b0;
    }

    *M(c, ldc, 0, 0) += c00;
    *M(c, ldc, 1, 0) += c10;
    *M(c, ldc, 2, 0) += c20;
    *M(c, ldc, 3, 0) += c30;
}

// Same as vec version, but works on every alignment
void scalar_kernel_1x4(
    pointer __restrict a, pointer __restrict b, 
    pointer __restrict c, size_t n, 
    size_t lda, size_t ldb, size_t ldc
)
{
    number_t c00 = 0, c01 = 0, c02 = 0, c03 = 0;
    number_t b0, b1, b2, b3;
    number_t a0;

    for (size_t i = 0; i < n; i++) {
        // c00 = dot(a0x, bx0), c01 = dot(a0x, bx1), ...
        b0 = *M(b, ldb, i, 0);
        b1 = *M(b, ldb, i, 1);
        b2 = *M(b, ldb, i, 2);
        b3 = *M(b, ldb, i, 3);
        
        a0 = *M(a, lda, 0, i);

        c00 += a0 * b0;
        c01 += a0 * b1;
        c02 += a0 * b2;
        c03 += a0 * b3;
    }

    *M(c, ldc, 0, 0) += c00;
    *M(c, ldc, 0, 1) += c01;
    *M(c, ldc, 0, 2) += c02;
    *M(c, ldc, 0, 3) += c03;
}

// This doesn't work on some cases when ldb is not a multiple of 4
void scalar_kernel_vec1x4(
    pointer __restrict a, pointer __restrict b, 
    pointer __restrict c, size_t n, 
    size_t lda, size_t ldb, size_t ldc
)
{
    vec4f_t va, vb, vc;
    vc.v = _mm_setzero_ps();

    for (size_t i = 0; i < n; i++) {
        vb.v = _mm_load_ps(M(b, ldb, i, 0));
        
        // a0 = *M(a, lda, 0, i);
        va.v = _mm_set1_ps(*M(a, lda, 0, i));
        vc.v = _mm_add_ps(vc.v, _mm_mul_ps(va.v, vb.v));
    }

    // *M(c, ldc, 0, 0) += c00;
    // *M(c, ldc, 0, 1) += c01;
    vc.v = _mm_add_ps(_mm_load_ps(M(c, ldc, 0, 0)), vc.v);
    _mm_store_ps(M(c, ldc, 0, 0), vc.v);
}

// Expects a[0:4, 0:n], b[0:n, 0:4] and c[0:4, 0:4] where [a:b], b is not included
// Will work on unaligned memory
void add_kernel_dot_4x4(
    pointer __restrict a, pointer __restrict b, 
    pointer __restrict c, size_t n, 
    size_t lda, size_t ldb, size_t ldc
)
{
    number_t c00 = 0, c01 = 0, c02 = 0, c03 = 0,
            c10 = 0, c11 = 0, c12 = 0, c13 = 0,
            c20 = 0, c21 = 0, c22 = 0, c23 = 0,
            c30 = 0, c31 = 0, c32 = 0, c33 = 0;
    number_t a0, a1, a2, a3;
    number_t b0, b1, b2, b3;

    // Take the whole row of a and calc dot product with the whole column of b
    for (size_t i = 0; i < n; i++) {
        // c00 = dot(a0x, bx0), c01 = dot(a0x, bx1), ...
        b0 = *M(b, ldb, i, 0);
        b1 = *M(b, ldb, i, 1);
        b2 = *M(b, ldb, i, 2);
        b3 = *M(b, ldb, i, 3);
        
        a0 = *M(a, lda, 0, i);
        a1 = *M(a, lda, 1, i);
        a2 = *M(a, lda, 2, i);
        a3 = *M(a, lda, 3, i);

        c00 += a0 * b0;
        c01 += a0 * b1;
        c02 += a0 * b2;
        c03 += a0 * b3;

        c10 += a1 * b0;
        c11 += a1 * b1;
        c12 += a1 * b2;
        c13 += a1 * b3;

        c20 += a2 * b0;
        c21 += a2 * b1;
        c22 += a2 * b2;
        c23 += a2 * b3;

        c30 += a3 * b0;
        c31 += a3 * b1;
        c32 += a3 * b2;
        c33 += a3 * b3;
    }

    *M(c, ldc, 0, 0) += c00;
    *M(c, ldc, 0, 1) += c01;
    *M(c, ldc, 0, 2) += c02;
    *M(c, ldc, 0, 3) += c03;

    *M(c, ldc, 1, 0) += c10;
    *M(c, ldc, 1, 1) += c11;
    *M(c, ldc, 1, 2) += c12;
    *M(c, ldc, 1, 3) += c13;

    *M(c, ldc, 2, 0) += c20;
    *M(c, ldc, 2, 1) += c21;
    *M(c, ldc, 2, 2) += c22;
    *M(c, ldc, 2, 3) += c23;

    *M(c, ldc, 3, 0) += c30;
    *M(c, ldc, 3, 1) += c31;
    *M(c, ldc, 3, 2) += c32;
    *M(c, ldc, 3, 3) += c33;
}

void M_tiled_multiply(
    size_t m, size_t n, size_t k, pointer __restrict a, 
    pointer __restrict b, pointer __restrict c, 
    size_t lda, size_t ldb, size_t ldc
) noexcept
{
    // Should be sqrt(M), where M is size of the cache in bytes,
    // choosing 256*256 = 65356 as default
    constexpr auto tile_size = 256;
    
    // nxm, mxp
    for (int I = 0; I < m; I += tile_size) {
        for (int J = 0; J < k; J += tile_size) {
            for (int K = 0; K < n; K += tile_size) {
                
                // Multiply A(I:I+T, K:K+T) * B(K:K+T, J:J+T) = C(I:I+T, J:J+T)
                for (int i = I; i < std::min(I+tile_size, int(m)); i++) {
                    for (int j = J; j < std::min(J+tile_size, int(k)); j++) {
                        number_t sum = 0;

                        for (int l = K; l < std::min(K+tile_size, int(n)); l++) {
                            sum += *M(a, lda, i, l) * *M(b, ldb, l, j);
                        }
                        *M(c, ldc, i, j) = sum;
                    }
                }
            }
        }
    }
}

void packA(number_t* pack, number_t* A, size_t m, size_t n, size_t lda) {
    // Assumes m * n is a multiple of 32
    // number_t* pack = utils::aligned_alloc<32, number_t>(m * n);
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            pack[i * n + j] = A[i * lda + j];
            // *pack++ = A[i * lda + j]
        }
    }
}

// Calculates C = A * B, where each matrix is row-major and 
// dimensions are: A: m x n, B: n x k, C: m x k, 
// in standard case lda = n, ldb = k, ldc = k
// 
// kernel calculates mini matrix multiplication (using 4x4 size)
template <size_t kernel_size>
void cpu_ops::M_inner_kernel(
    size_t m, size_t n, size_t k, pointer __restrict A, 
    pointer __restrict B, pointer __restrict C, 
    size_t lda, size_t ldb, size_t ldc, 
    cpu_ops::func_kernel_dot_t kernel,
    func_scalar_kernel_t kernel_1xN,
    func_scalar_kernel_t kernel_Nx1
) noexcept
{
    constexpr auto MR = kernel_size, NR = kernel_size;

    // number_t* aPacked = utils::aligned_alloc<32, number_t>(MC * NC);
    // number_t* bPacked = utils::aligned_alloc<32, number_t>(NC * KC);

    // utils::print_matrix(A, m, n);
    // utils::print_matrix(B, n, k);
    // utils::print_matrix(C, m, k);
    

    for (size_t K0 = 0; K0 < k; K0 += KC) {
        const size_t kc = std::min(KC, k - K0);
        for (size_t N0 = 0; N0 < n; N0 += NC) {
            const size_t nc = std::min(NC, n - N0);

            // Pack B(N0:N0+nc, K0:K0+kc) into bPacked
            packA(bPacked, &B[N0 * ldb + K0], nc, kc, ldb);

            // std::cout << "BBB: ";
            // utils::print_matrix(bPacked, nc, kc);

            for (size_t M0 = 0; M0 < m; M0 += MC) {
                size_t mc = std::min(MC, m - M0);
        
                // Pack A(M0:M0+mc, N0:N0+nc) to aPacked
                packA(aPacked, &A[M0 * lda + N0], mc, nc, lda);

                // std::cout << "AAA: ";
                // utils::print_matrix(aPacked, mc, nc);

                // If mc is not a multiple of MR, we must 
                // handle the rest of the rows, same for kc
                // so this simply get the 'multiple of MR' part from mc
                const size_t mc_main = (mc / MR) * MR;
                const size_t kc_main = (kc / NR) * NR;

                 // Multiply A(M0:M0+mc, N0:N0+nc) * B(N0:N0+nc, K0:K0+kc) = C(M0:M0+mc, K0:K0+kc)
                for (size_t ir = 0; ir < mc_main; ir += MR) {
                    for (size_t jr = 0; jr < kc_main; jr += NR) {
                        
                        // submatrix of a(M0:M0+mc, N0:N0+nc), lda_sub = nc
                        pointer a_sub = &aPacked[ir * nc + 0];
                        // submatrix of b(N0:N0+nc, K0:K0+kc), ldb_sub = kc
                        pointer b_sub = &bPacked[0 * kc + jr];
                        // c matrix
                        pointer c_sub = &C[jr + K0 + (M0 + ir) * ldc];

                        // n (inner-product lenght) is nc
                        kernel(
                            a_sub, b_sub, c_sub,
                            nc, nc, kc, ldc
                        );

                        // utils::print_matrix(C, m, k);
                    }

                    // Handle rest of the columns with 4x1 kernel
                    for (size_t j = kc_main; j < kc; j++) {
                        kernel_Nx1(
                            &aPacked[ir * nc + 0],
                            &bPacked[0 * kc + j],
                            &C[j + K0 + (M0 + ir) * ldc],
                            nc, nc, kc, ldc
                        );
                    }
                }

                // Bottom-edge
                for (size_t i = mc_main; i < mc; i++) {
                    for (size_t jr = 0; jr < kc_main; jr+=NR) {
                        kernel_1xN(
                            &aPacked[i * nc + 0],
                            &bPacked[0 * kc + jr],
                            &C[jr + K0 + (M0 + i) * ldc],
                            nc, nc, kc, ldc
                        );
                    }

                    // Corner
                    for (size_t j = kc_main; j < kc; j++) {
                        number_t sum = 0;
                        for (size_t p = 0; p < nc; p++) {
                            sum += aPacked[i * nc + p] * bPacked[p * kc + j];
                        }
                        C[(M0 + i) * ldc + (j + K0)] += sum;
                    }
                }

                // Handle rest of the rows with 1x4 kernel
                // and handle the remaining corner block with naive impl
            }
        }
    }

    // std::free(aPacked);
    // std::free(bPacked);
}

// I know this is pointless, since if the cpu doesn't support avx2,
// it won't compile (i think), but I want to keep the code clean
template <cpu_ops::matmul_type type>
void cpu_ops::M_impl_matumul(
    pointer __restrict a, size_t lda, 
    pointer __restrict b, size_t ldb,
    pointer __restrict c, size_t ldc,
    size_t m, size_t n, size_t k
) noexcept
{
    // M_inner_kernel<4>(
    //     m, n, k, a, b, c, lda, ldb, ldc, 
    //     M_add_kernel_dot_4x4, 
    //     scalar_kernel_1x4,
    //     scalar_kernel_4x1
    // );
    M_inner_kernel<8>(
        m, n, k, a, b, c, lda, ldb, ldc, 
        M_add_kernel_dot_8x8, 
        scalar_kernel_1x8,
        scalar_kernel_8x1
    );
}


// Get the result tensor, based on the a's and b's operation flags.
// Asserts same size of a and b, then tries to borrow a's or b's buffers
inline tensor_t cpu_ops::M_get_res_tensor(tensor_t& a, tensor_t& b)
{
    // Assert same size
    M_assert_tensor_same_size(a, b);

    // Try to borrow buffers, to avoid redundand memory allocation
    tensor_t res = ops_tensor::M_get_vector_like(a, b);

    // if in chained operation, will use that fact in the function above
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

    auto res = M_get_res_tensor(a, b);
    // calculate the result

    const auto size = res.size();
    // if (M_should_lanuch_threads(size)) 
    // {
    //     // dispatch threads to do the work in parallel
    //     int offset_size = size / m_ncores;
    //     int offset = 0;
    //     auto a_data = a.data();
    //     auto b_data = b.data();
    //     auto res_data = res.data();

    //     for (int i = 0; i < m_ncores; i++, offset += offset_size) {
    //         size_t patch_size = i == m_ncores - 1 ? size - offset : offset_size;
    //         m_threads.push_back(
    //             std::thread(
    //                 func, a_data + offset, b_data + offset, 
    //                 res_data + offset, patch_size, n, m
    //             ));
    //     }

    //     M_join_threads();
    // }
    // else
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
tensor_rref_t cpu_ops::mat_mul(tensor_t a, tensor_t b)
{
    M_assert_tensor_dim_mat_mul(a, b);

    const size_t m_rows = a.dim()[0];
    const size_t n_cols = a.dim()[1];
    const size_t k_cols = b.dim()[1];

    auto res = ops_tensor({m_rows, k_cols});
    
    M_impl_matumul<cpu_ops::matmul_type::see>(
        a.data(), n_cols,
        b.data(), k_cols,
        res.data(), k_cols,
        m_rows, n_cols, k_cols
    );

    return std::move(res);
}

tensor_rref_t cpu_ops::transpose(tensor_t a)
{
    return std::move(a);
}

END_BLUST_NAMESPACE