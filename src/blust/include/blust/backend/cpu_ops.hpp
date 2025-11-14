#pragma once

#include <immintrin.h>
#include <cstring>
#include <string.h>
#include <thread>
#include <vector>

#include "operations.hpp"
#include <blust/error.hpp>

START_BLUST_NAMESPACE

class cpu_ops : public operations
{
    friend class cpu_backend;

    typedef tensor_t::pointer pointer;
    typedef void(*func_vector_t)(pointer, pointer, pointer, size_t, number_t, number_t);
    typedef void(*func_kernel_dot_t)(
        pointer __restrict, pointer __restrict, 
        pointer __restrict, size_t, size_t, size_t, size_t
    );
    typedef void(*func_scalar_kernel_t)(
        pointer __restrict, pointer __restrict, 
        pointer __restrict, size_t, size_t, size_t, size_t
    );

    // Preformns c = a * n + b * m
    static void M_impl_add(
        pointer a, pointer b, pointer c, 
        size_t size, number_t n, number_t m
    ) noexcept(true);

    static void M_impl_hadamard(
        pointer a, pointer b, pointer c, 
        size_t size, number_t, number_t
    ) noexcept(true);

    template <size_t kernel_r, size_t kernel_c>
    void M_inner_kernel(
        size_t m, size_t n, size_t k, pointer __restrict a, 
        pointer __restrict b, pointer __restrict c, 
        size_t lda, size_t ldb, size_t ldc, 
        func_kernel_dot_t kernel,
        func_scalar_kernel_t kernel_1xN,
        func_scalar_kernel_t kernel_Nx1
    ) noexcept(true);

    enum class matmul_type { avx2, see };

    template <matmul_type type = matmul_type::avx2>
    void M_impl_matumul(
        pointer __restrict a, size_t lda, 
        pointer __restrict b, size_t ldb,
        pointer __restrict c, size_t ldc,
        size_t n, size_t m, size_t k
    ) noexcept(true);

    // Calls M_add with these parameters
    inline tensor_t M_perform_vector_like(
        tensor_t& a, tensor_t& b, number_t n, number_t m,
        func_vector_t func, bool allocate
    );

    static inline tensor_t M_get_res_tensor(tensor_t& a, tensor_t& b);

    // Checks if the size is big enough to launch threads
    inline bool M_should_lanuch_threads(size_t size) noexcept {
        return m_ncores > 1 && size > 5e5;
    }

    // Joins all the threads
    inline void M_join_threads() noexcept
    {
        for (auto& t : m_threads) 
            if (t.joinable())
                t.join();
    }

    int m_ncores{1};
    std::vector<std::thread> m_threads;
    number_t *aPacked{nullptr}, *bPacked{nullptr};

public:
    static constexpr size_t MC = 128, NC = 128, KC = 256; // L2 cache blocking

    cpu_ops(int n_threads = 1);
    ~cpu_ops();

    tensor_rref_t add(tensor_t, tensor_t, bool allocate = true) override;
    tensor_rref_t sub(tensor_t, tensor_t, bool allocate = true) override;
    tensor_rref_t mul(tensor_t, number_t, bool allocate = true) override;
    tensor_rref_t div(tensor_t, number_t, bool allocate = true) override;

    tensor_rref_t hadamard(tensor_t, tensor_t, bool allocate = true) override;
    tensor_rref_t mat_mul(tensor_t, tensor_t) override;
    tensor_rref_t transpose(tensor_t) override;
};

END_BLUST_NAMESPACE