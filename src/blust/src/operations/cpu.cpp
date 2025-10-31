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
    while (size--) {
        (*c_data++) = (*a_data++) * n + (*b_data++) * m;
    }

    // size_t i = 0;
    // if (size >= 4 * 8)
    // {
    //     // SIMD version with 4 avx2 vectors
    //     vec8f_t va0, va1, va2, va3,
    //             vb0, vb1, vb2, vb3,
    //             vc0, vc1, vc2, vc3,
    //             vn, vm;

    //     vn.v = _mm256_set1_ps(n);
    //     vm.v = _mm256_set1_ps(m);

    //     for (; i < size; i += 4 * 8)
    //     {
    //         if (size - i < 4 * 8)
    //             break;

    //         va0.v = _mm256_load_ps(a_data + i);
    //         va1.v = _mm256_load_ps(a_data + i + 8);
    //         va2.v = _mm256_load_ps(a_data + i + 16);
    //         va3.v = _mm256_load_ps(a_data + i + 24);

    //         vb0.v = _mm256_load_ps(b_data + i);
    //         vb1.v = _mm256_load_ps(b_data + i + 8);
    //         vb2.v = _mm256_load_ps(b_data + i + 16);
    //         vb3.v = _mm256_load_ps(b_data + i + 24);

    //         vc0.v = _mm256_add_ps(_mm256_mul_ps(va0.v, vn.v), _mm256_mul_ps(vb0.v, vm.v));
    //         vc1.v = _mm256_add_ps(_mm256_mul_ps(va1.v, vn.v), _mm256_mul_ps(vb1.v, vm.v));
    //         vc2.v = _mm256_add_ps(_mm256_mul_ps(va2.v, vn.v), _mm256_mul_ps(vb2.v, vm.v));
    //         vc3.v = _mm256_add_ps(_mm256_mul_ps(va3.v, vn.v), _mm256_mul_ps(vb3.v, vm.v));

    //         _mm256_store_ps(c_data + i, vc0.v);
    //         _mm256_store_ps(c_data + i + 8, vc1.v);
    //         _mm256_store_ps(c_data + i + 16, vc2.v);
    //         _mm256_store_ps(c_data + i + 24, vc3.v);
    //     }
    // }

    // // Add the rest of the elements
    // for(; i < size; i++) {
    //     (*c_data++) += (*a_data++) * n + (*b_data++) * m;
    // }    
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

    // with -03 and -mavx this is faster
    while (size--) {
        (*c_data++) = (*a_data++) * (*b_data++);
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
    vec8f_t 
        va0, va1, va2, va3, va4, va5, va6, va7, // 8 row with 8 elements of equal value
        vb, // column with 8 elements
        vc0, vc1, vc2, vc3, vc4, vc5, vc6, vc7;  // 8 rows of c (with 8 elements)

    vc0.v = _mm256_loadu_ps(M(c, ldc, 0, 0));
    vc1.v = _mm256_loadu_ps(M(c, ldc, 1, 0));
    vc2.v = _mm256_loadu_ps(M(c, ldc, 2, 0));
    vc3.v = _mm256_loadu_ps(M(c, ldc, 3, 0));
    vc4.v = _mm256_loadu_ps(M(c, ldc, 4, 0));
    vc5.v = _mm256_loadu_ps(M(c, ldc, 5, 0));
    vc6.v = _mm256_loadu_ps(M(c, ldc, 6, 0));
    vc7.v = _mm256_loadu_ps(M(c, ldc, 7, 0));

    for(size_t i = 0; i < n; i++)
    {
        // Load the row of a
        va0.v = _mm256_set1_ps(*M(a, lda, 0, i));
        va1.v = _mm256_set1_ps(*M(a, lda, 1, i));
        va2.v = _mm256_set1_ps(*M(a, lda, 2, i));
        va3.v = _mm256_set1_ps(*M(a, lda, 3, i));
        va4.v = _mm256_set1_ps(*M(a, lda, 4, i));
        va5.v = _mm256_set1_ps(*M(a, lda, 5, i));
        va6.v = _mm256_set1_ps(*M(a, lda, 6, i));
        va7.v = _mm256_set1_ps(*M(a, lda, 7, i));
        
        // Load the column of b
        vb.v = _mm256_loadu_ps(M(b, ldb, i, 0));
        // b0, b1, b2, b3

        vc0.v = _mm256_add_ps(vc0.v, _mm256_mul_ps(va0.v, vb.v));
        // c00 = a0 * b0, c01 = a0 * b1, c02 = a0 * b2, c03 = a0 * b3

        // same for the rest of the rows
        vc1.v = _mm256_add_ps(vc1.v, _mm256_mul_ps(va1.v, vb.v));
        vc2.v = _mm256_add_ps(vc2.v, _mm256_mul_ps(va2.v, vb.v));
        vc3.v = _mm256_add_ps(vc3.v, _mm256_mul_ps(va3.v, vb.v));
        vc4.v = _mm256_add_ps(vc4.v, _mm256_mul_ps(va4.v, vb.v));
        vc5.v = _mm256_add_ps(vc5.v, _mm256_mul_ps(va5.v, vb.v));
        vc6.v = _mm256_add_ps(vc6.v, _mm256_mul_ps(va6.v, vb.v));
        vc7.v = _mm256_add_ps(vc7.v, _mm256_mul_ps(va7.v, vb.v));
    }

    // Store the result (8 rows of c with 8 elements)
    _mm256_storeu_ps(M(c, ldc, 0, 0), vc0.v);
    _mm256_storeu_ps(M(c, ldc, 1, 0), vc1.v);
    _mm256_storeu_ps(M(c, ldc, 2, 0), vc2.v);
    _mm256_storeu_ps(M(c, ldc, 3, 0), vc3.v);
    _mm256_storeu_ps(M(c, ldc, 4, 0), vc4.v);
    _mm256_storeu_ps(M(c, ldc, 5, 0), vc5.v);
    _mm256_storeu_ps(M(c, ldc, 6, 0), vc6.v);
    _mm256_storeu_ps(M(c, ldc, 7, 0), vc7.v);
}

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
        va0, va1, va2, va3, // 4 row with 4 elements of equal value
        vb, // column with 4 elements
        vc0, vc1, vc2, vc3;  // 4 rows of c (with 4 elements)

    vc0.v = _mm_load_ps(M(c, ldc, 0, 0));
    vc1.v = _mm_load_ps(M(c, ldc, 1, 0));
    vc2.v = _mm_load_ps(M(c, ldc, 2, 0));
    vc3.v = _mm_load_ps(M(c, ldc, 3, 0));

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

        vc0.v = _mm_add_ps(vc0.v, _mm_mul_ps(va0.v, vb.v));
        // c00 = a0 * b0, c01 = a0 * b1, c02 = a0 * b2, c03 = a0 * b3

        // same for the rest of the rows
        vc1.v = _mm_add_ps(vc1.v, _mm_mul_ps(va1.v, vb.v));
        vc2.v = _mm_add_ps(vc2.v, _mm_mul_ps(va2.v, vb.v));
        vc3.v = _mm_add_ps(vc3.v, _mm_mul_ps(va3.v, vb.v));
    }

    // Store the result (4 rows of c with
    _mm_store_ps(M(c, ldc, 0, 0), vc0.v);
    _mm_store_ps(M(c, ldc, 1, 0), vc1.v);
    _mm_store_ps(M(c, ldc, 2, 0), vc2.v);
    _mm_store_ps(M(c, ldc, 3, 0), vc3.v);
}

void M_add_kernel_dot_2x2(
    pointer __restrict a, pointer __restrict b, 
    pointer __restrict c, size_t n, 
    size_t lda, size_t ldb, size_t ldc
) 
{
    number_t c00 = 0, c01 = 0, c10 = 0, c11 = 0;

    // Take the whole row of a and calc dot product with the whole column of b
    for (size_t i = 0; i < n; i++) {
        // c00 = dot(a0x, bx0)
        c00 += *M(a, lda, 0, i) * *M(b, ldb, i, 0);
        c01 += *M(a, lda, 0, i) * *M(b, ldb, i, 1);
        c10 += *M(a, lda, 1, i) * *M(b, ldb, i, 0);
        c11 += *M(a, lda, 1, i) * *M(b, ldb, i, 1);
    }

    *M(c, ldc, 0, 0) = c00;
    *M(c, ldc, 0, 1) = c01;
    *M(c, ldc, 1, 0) = c10;
    *M(c, ldc, 1, 1) = c11;
}

template <size_t kernel_size>
void cpu_ops::M_inner_kernel(
    size_t m, size_t n, size_t k, pointer __restrict a, 
    pointer __restrict b, pointer __restrict c, 
    size_t lda, size_t ldb, size_t ldc, cpu_ops::func_kernel_dot_t kernel
) noexcept
{
    size_t cols = 0, rows = 0;

    for (; rows < m; rows+=kernel_size) // loop over the columns of c (and b's)
    {
        if (rows + kernel_size > m) 
            break;
        
        for (cols = 0; cols < k; cols+=kernel_size) // loop over the rows of c (and a's)
        {
            if (cols + kernel_size > k) 
                break;

            // Calculate the dot product of a row of A
            // and a column of B
            kernel(
                M(a, lda, rows, 0),
                M(b, ldb, 0, cols),
                M(c, ldc, rows, cols),
                n, lda, ldb, ldc
            );
        }
    }

    // loop over the rest of rows
    for (; rows < m; rows++)
    {
        for (cols = 0; cols < k; cols++)
        {
            number_t sum = 0;
            for (size_t l = 0; l < n; l++)
                sum += *M(a, lda, rows, l) * *M(b, ldb, l, cols);
            (*M(c, ldc, rows, cols)) = sum;
        }
    }

    // loop over the rest of columns
    size_t rest = k % kernel_size;
    if (rest == 0)
        return;
    
    rest = k - rest;
    for (rows = 0; rows < m; rows++)
    {
        for (cols = rest; cols < k; cols++)
        {
            number_t sum = 0;
            for (size_t l = 0; l < n; l++)
                sum += *M(a, lda, rows, l) * *M(b, ldb, l, cols);
            (*M(c, ldc, rows, cols)) = sum;
        }
    }
}

template <size_t kernel_size>
void cpu_ops::M_calc_kernel_dot(
    pointer __restrict a, pointer __restrict b, 
    pointer __restrict c, cpu_ops::func_kernel_dot_t kernel,
    size_t m, size_t n, size_t k, size_t lda, size_t ldb, size_t ldc
) noexcept
{
    M_inner_kernel<2>(
        m, n, k, a, b, c, lda, ldb, ldc, M_add_kernel_dot_2x2
    );

    // M_inner_kernel<kernel_size>(
    //     m, n, k, a, b, c, lda, ldb, ldc, kernel
    // );

    // constexpr size_t pack_size_m = 256, pack_size_n = 128;

    // size_t packed_m, packed_n;

    // for (size_t i = 0; i < n; i += pack_size_n)
    // {
    //     packed_n = std::min(pack_size_n, n - i);
    //     for (size_t j = 0; j < m; j += pack_size_m)
    //     {
    //         packed_m = std::min(pack_size_m, m - j);
    //         M_inner_kernel<kernel_size>(
    //             packed_m, packed_n, k,
    //             M(a, lda, j, i), M(b, ldb, i, 0), 
    //             M(c, ldc, j, 0),
    //             lda, ldb, ldc, kernel
    //         );
    //     }
    // }
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
    M_inner_kernel<2>(m, n, k, a, b, c, lda, ldb, ldc, M_add_kernel_dot_2x2);
    // if constexpr (type == matmul_type::avx2)
    //     M_calc_kernel_dot<8>(a, b, c, M_add_kernel_dot_8x8, m, n, k, lda, ldb, ldc);
    // else
    //     M_calc_kernel_dot<4>(a, b, c, M_add_kernel_dot_4x4, m, n, k, lda, ldb, ldc);
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
    func_vector_t func
)
{
    BLUST_ASSERT(a.dim()==b.dim());

    auto res = M_get_res_tensor(a, b);
    // calculate the result

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
tensor_rref_t cpu_ops::add(tensor_t a, tensor_t b) 
{
    return M_perform_vector_like(a, b, 1.0, 1.0, M_impl_add);
}

/**
 * @brief Perform substaction (a - b) and return the result
 */
tensor_rref_t cpu_ops::sub(tensor_t a, tensor_t b) 
{
    return M_perform_vector_like(a, b, 1.0, -1.0, M_impl_add);
}

/**
 * @brief Caluculate Ri = Ai * b (see hadamard for element-wise multiplication)
 */
tensor_rref_t cpu_ops::mul(tensor_t a, number_t b) 
{
    return M_perform_vector_like(a, a, b, 0.0, M_impl_add); // c = a * b + a * 0
}

/**
 * @brief Calculate Ri = Ai / b
 */
tensor_rref_t cpu_ops::div(tensor_t a, number_t b) 
{
    return M_perform_vector_like(a, a, 1 / b, 0.0, M_impl_add);
}

/**
 * @brief Get the hadamard product: Ci = Ai * Bi
 */
tensor_rref_t cpu_ops::hadamard(tensor_t a, tensor_t b)
{
    return M_perform_vector_like(a, b, 0, 0, M_impl_hadamard);
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

    const int m_rows = a.dim()[0];
    const int n_cols = a.dim()[1];
    const int k_cols = b.dim()[1];

    auto res = ops_tensor({m_rows, k_cols});
    
    M_impl_matumul<cpu_ops::matmul_type::see>(
        a.data(), n_cols,
        b.data(), k_cols,
        res.data(), k_cols,
        m_rows, n_cols, k_cols
    );

    return std::move(res);
}

END_BLUST_NAMESPACE