#include <blust/backend/cpu_ops.hpp>

#include <sys/time.h>

START_BLUST_NAMESPACE

typedef operations::tensor_t tensor_t;
typedef tensor_t::pointer pointer;

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

    // if (size >= 4)
    // {
    //     vec8f_t va, vb, vc, vn, vm;

    //     // Must be aligned
    //     alignas(tensor::alignment) number_t nvec[8] = {n, n, n, n, n, n, n, n};
    //     alignas(tensor::alignment) number_t mvec[8] = {m, m, m, m, m, m, m, m};

    //     vn.v = _mm256_load_ps(nvec);
    //     vm.v = _mm256_load_ps(mvec);

    //     for (; i < size; i += 8)
    //     {
    //         va.v = _mm256_load_ps(a_data + i);
    //         vb.v = _mm256_load_ps(b_data + i);
    //         vc.v = _mm256_add_ps(_mm256_mul_ps(va.v, vn.v), _mm256_mul_ps(vb.v, vm.v));
    //         _mm256_store_ps(c_data + i, vc.v);
    //     }
    // }

    // // Add the rest of the elements
    // for (;i < size; i++) {
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

void cpu_ops::M_add_dot_4x4(
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
        vc0, vc1, vc2, vc3; // 4 rows of c (with 4 elements)
    
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

        vc0.v = _mm_add_ps(vc0.v, _mm_mul_ps(va0.v, vb.v));\
        // c00 = a0 * b0, c01 = a0 * b1, c02 = a0 * b2, c03 = a0 * b3

        // same for the rest of the rows
        vc1.v = _mm_add_ps(vc1.v, _mm_mul_ps(va1.v, vb.v));
        vc2.v = _mm_add_ps(vc2.v, _mm_mul_ps(va2.v, vb.v));
        vc3.v = _mm_add_ps(vc3.v, _mm_mul_ps(va3.v, vb.v));
    }

    // Store the result (4 rows of c with 4 elements)
    _mm_store_ps(M(c, ldc, 0, 0), vc0.v);
    _mm_store_ps(M(c, ldc, 1, 0), vc1.v);
    _mm_store_ps(M(c, ldc, 2, 0), vc2.v);
    _mm_store_ps(M(c, ldc, 3, 0), vc3.v);
}

void cpu_ops::M_impl_matumul(
    pointer __restrict a, size_t lda, 
    pointer __restrict b, size_t ldb,
    pointer __restrict c, size_t ldc,
    size_t m, size_t n, size_t k
) noexcept
{
    size_t i = 0, j = 0;

    for (; i < k; i+=4) // loop over the columns of c (and b's)
    {
        if (i + 4 > k) 
            break;

        for (j = 0; j < m; j+=4) // loop over the rows of c (and a's)
        {
            if (j + 4 > m) 
                break;

            // Calculate the dot product of the ith row of A
            // and the jth column of B
            M_add_dot_4x4(
                M(a, lda, j, 0),
                M(b, ldb, 0, i),
                M(c, ldc, j, i),
                n, lda, ldb, ldc
            );
        }
    }

    // loop over the rest of the columns
    for (; i < k; i++)
    {
        for (j = 0; j < m; j++)
        {
            number_t sum = 0;
            for (size_t l = 0; l < n; l++)
                sum += *M(a, lda, j, l) * *M(b, ldb, l, i);
            (*M(c, ldc, j, i)) = sum;
        }
    }
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
            m_threads.push_back(std::thread(
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
tensor_t cpu_ops::add(tensor_t a, tensor_t b) 
{
    return M_perform_vector_like(a, b, 1.0, 1.0, M_impl_add);
}

/**
 * @brief Perform substaction (a - b) and return the result
 */
tensor_t cpu_ops::sub(tensor_t a, tensor_t b) 
{
    return M_perform_vector_like(a, b, 1.0, -1.0, M_impl_add);
}

/**
 * @brief Caluculate Ri = Ai * b (see hadamard for element-wise multiplication)
 */
tensor_t cpu_ops::mul(tensor_t a, number_t b) 
{
    return M_perform_vector_like(a, a, b, 0.0, M_impl_add); // c = a * b + a * 0
}

/**
 * @brief Calculate Ri = Ai / b
 */
tensor_t cpu_ops::div(tensor_t a, number_t b) 
{
    return M_perform_vector_like(a, a, 1 / b, 0.0, M_impl_add);
}

/**
 * @brief Get the hadamard product: Ci = Ai * Bi
 */
tensor_t cpu_ops::hadamard(tensor_t a, tensor_t b)
{
    return M_perform_vector_like(a, b, 0, 0, M_impl_hadamard);
}

/**
 * @brief Perform matrix multiplication, and return the result
 * @param a the first matrix, with dimensions m x n and in a column-major order
 * @param b the second matrix, with dimensions n x k and in a column-major order
 * @return the result matrix, with dimensions m x k and in a column-major order
 */
tensor_t cpu_ops::mat_mul(tensor_t a, tensor_t b)
{
    M_assert_tensor_dim_mat_mul(a, b);

    const int m_rows = a.dim()[0];
    const int n_cols = a.dim()[1];
    const int k_cols = b.dim()[1];

    auto res = ops_tensor({m_rows, k_cols});
    
    M_impl_matumul(
        a.data(), n_cols,
        b.data(), k_cols,
        res.data(), k_cols,
        m_rows, n_cols, k_cols
    );

    return res;
}

END_BLUST_NAMESPACE