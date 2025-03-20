#include <blust/backend/cpu_ops.hpp>

#include <sys/time.h>

START_BLUST_NAMESPACE

typedef operations::tensor_t tensor_t;
typedef tensor_t::pointer pointer;

typedef union {
    __m128 v;
    float d[4];
} vec4f_t;


/**
 * @brief Performs c = a * n + b * m, all pointers should be 16-byte aligned
 */
void cpu_ops::M_add(
    pointer a_data, pointer b_data, 
    pointer c_data, size_t size, 
    number_t n, number_t m
) noexcept
{
    size_t i = 0;

    if (size >= 4)
    {
        vec4f_t va, vb, vc, vn, vm;

        // Must be aligned
        alignas(16) number_t nvec[4] = {n, n, n, n};
        alignas(16) number_t mvec[4] = {m, m, m, m};

        vn.v = _mm_load_ps(nvec);
        vm.v = _mm_load_ps(mvec);

        for (; i < size; i += 4)
        {
            va.v = _mm_load_ps(a_data + i);
            vb.v = _mm_load_ps(b_data + i);
            vc.v = _mm_add_ps(_mm_mul_ps(va.v, vn.v), _mm_mul_ps(vb.v, vm.v));
            _mm_store_ps(c_data + i, vc.v);
        }
    }

    // Add the rest of the elements
    for (;i < size; i++, a_data++, b_data++, c_data++) {
        *c_data += *a_data * n + *b_data * m;
    }
}

/**
 * @brief Add two tensors and return the result
 */
tensor_t cpu_ops::add(tensor_t a, tensor_t b)
{
    M_assert_tensor_same_size(a, b);
    tensor_t res(tensor::aligned_alloc(a.size()), a.layout());

    // Perform a tiled addition of the 2 tensors
    struct timeval start, finish;
	double gflops = 2.0 * a.size() * 1e-9;

    gettimeofday(&start, NULL);
    M_add(a.data(), b.data(), res.data(), res.size(), 1.0, 1.0);
    gettimeofday(&finish, NULL);

    double duration = ((double)(finish.tv_sec-start.tv_sec)*1000000 + (double)(finish.tv_usec-start.tv_usec)) / 1000000;
	printf("add took %f seconds GFLOPS : %f\n",duration,gflops/duration);

    return res;
}

/**
 * @brief Perform substaction (a - b) and return the result
 */
tensor_t cpu_ops::sub(tensor_t a, tensor_t b)
{
    M_assert_tensor_same_size(a, b);
    tensor_t res(tensor::aligned_alloc(a.size()), a.layout());
    M_add(a.data(), b.data(), res.data(), res.size(), 1.0, -1.0);
    return res;
}

/**
 * @brief Caluculate Ri = Ai * b
 */
tensor_t cpu_ops::mul(tensor_t a, number_t b)
{
    tensor_t res(tensor::aligned_alloc(a.size()), a.layout());
    M_add(a.data(), a.data(), res.data(), res.size(), 0.0, b);
    return res;
}

/**
 * @brief Calculate Ri = Ai / b
 */
tensor_t cpu_ops::div(tensor_t a, number_t b)
{
    tensor_t res(tensor::aligned_alloc(a.size()), a.layout());
    M_add(a.data(), a.data(), res.data(), res.size(), 0.0, 1 / b);
    return res;
}

/**
 * @brief Get the hadamard product: Ci = Ai * Bi
 */
tensor_t cpu_ops::hadamard(tensor_t a, tensor_t b)
{
    M_assert_tensor_same_size(a, b);

    const auto size = a.size();
    tensor_t res(tensor::aligned_alloc(size), a.layout());

    // Perform a hadamard product of the 2 tensors
    // Using SIMD instructions
    
    auto a_data = a.data();
    auto b_data = b.data();
    auto c_data = res.data();

    size_t i = 0;
    if (size >= 4)
    {
        vec4f_t va, vb, vc;

        for (; i < size; i += 4)
        {
            va.v = _mm_load_ps(a_data + i);
            vb.v = _mm_load_ps(b_data + i);
            vc.v = _mm_mul_ps(va.v, vb.v);
            _mm_store_ps(c_data + i, vc.v);
        }
    }

    // Multiply the rest of the elements
    for (;i < size; i++, a_data++, b_data++, c_data++) {
        *c_data += *a_data * *b_data;
    }

    return res;
}

/**
 * @brief Perform matrix multiplication, and return the result
 */
tensor_t cpu_ops::mat_mul(tensor_t a, tensor_t b)
{
    return a;
}

END_BLUST_NAMESPACE