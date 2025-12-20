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
    using ops_tensor_t = operations::ops_tensor_t;

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

    typedef void(*func_result_kernel_add_t)(
        const pointer __restrict kernel, pointer __restrict C, 
        size_t ldc
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
        size_t MC, size_t NC, size_t KC,
        func_kernel_dot_t kernel,
        func_scalar_kernel_t kernel_1xN,
        func_scalar_kernel_t kernel_Nx1
    ) noexcept(true);

    void M_impl_matumul(
        pointer __restrict a, size_t lda, 
        pointer __restrict b, size_t ldb,
        pointer __restrict c, size_t ldc,
        size_t n, size_t m, size_t k,
        size_t MC, size_t NC, size_t KC
    ) noexcept(true);

    // Calls M_add with these parameters
    inline void M_perform_vector_like(
        ops_tensor_t& a, ops_tensor_t& b, ops_tensor_t& res, number_t n, number_t m,
        func_vector_t func
    );

    // Checks if the size is big enough to launch threads
    inline bool M_should_lanuch_threads(size_t size) noexcept {
        return m_ncores > 1 && size > 5e5;
    }

    void M_realloc_packed(size_t MC, size_t KC, size_t NC) noexcept;


    // Joins all the threads
    inline void M_join_threads() noexcept
    {
        for (auto& t : m_threads) 
            if (t.joinable())
                t.join();
    }

    int m_ncores{1};
    size_t M_MC{0}, M_KC{0}, M_NC{0};
    std::vector<std::thread> m_threads;
    number_t *m_aPacked{nullptr}, *m_bPacked{nullptr};

public:
    cpu_ops(int n_threads = 1);
    ~cpu_ops();



    using operations::add;
    using operations::sub;
    using operations::mul;
    using operations::div;
    using operations::hadamard;
    using operations::mat_mul;
    using operations::transpose;

    void add(ops_tensor_t&, ops_tensor_t&, ops_tensor_t&) override;
    void sub(ops_tensor_t&, ops_tensor_t&, ops_tensor_t&) override;
    void mul(ops_tensor_t&, number_t, ops_tensor_t&) override;
    void div(ops_tensor_t&, number_t, ops_tensor_t&) override;

    void hadamard(ops_tensor_t&, ops_tensor_t&, ops_tensor_t&) override;
    void mat_mul(ops_tensor_t&, ops_tensor_t&, ops_tensor_t&, size_t MC, size_t KC, size_t NC);
    ops_tensor_t mat_mul(ops_tensor_t a, ops_tensor_t b, size_t MC, size_t KC, size_t NC) {
        M_assert_tensor_dim_mat_mul(a, b);
        ops_tensor res({ a.dim()[0], b.dim()[1] });
        mat_mul(a, b, res, MC, KC, NC);
        return std::move(res);
    }

    inline void mat_mul(ops_tensor_t& a, ops_tensor_t& b, ops_tensor_t& res) override {
        mat_mul(a, b, res, M_MC, M_KC, M_NC); 
    }
    void transpose(ops_tensor_t&, ops_tensor_t&) override;
};

END_BLUST_NAMESPACE