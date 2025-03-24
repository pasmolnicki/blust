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
constexpr void assume_aligned(pointer data) {
    data = (pointer) __builtin_assume_aligned(data, tensor::alignment);
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
 */
tensor_t cpu_ops::mat_mul(tensor_t a, tensor_t b)
{
    return a;
}

END_BLUST_NAMESPACE