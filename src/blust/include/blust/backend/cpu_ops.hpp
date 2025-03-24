#pragma once

#include <immintrin.h>
#include <cstring>
#include <string.h>
#include <thread>
#include <vector>
#include "operations.hpp"

START_BLUST_NAMESPACE

class cpu_ops : public operations
{
    typedef tensor_t::pointer pointer;
    typedef void(*func_vector_t)(pointer, pointer, pointer, size_t, number_t, number_t);

    // Preformns c = a * n + b * m
    static void M_impl_add(
        pointer a, pointer b, pointer c, 
        size_t size, number_t n, number_t m
    ) noexcept(true);

    static void M_impl_hadamard(
        pointer a, pointer b, pointer c, 
        size_t size, number_t, number_t
    ) noexcept(true);

    // Calls M_add with these parameters
    inline tensor_t M_perform_vector_like(
        tensor_t& a, tensor_t& b, number_t n, number_t m,
        func_vector_t func
    );

    static inline tensor_t M_get_res_tensor(tensor_t& a, tensor_t& b);

    // Checks if the size is big enough to launch threads
    inline bool M_should_lanuch_threads(size_t size) noexcept {
        return m_ncores > 1 && size > 2e5;
    }

    // Joins all the threads
    inline void M_join_threads() noexcept
    {
        for (auto& t : m_threads) 
            if (t.joinable())
                t.join();
    }

    int m_ncores;
    std::vector<std::thread> m_threads;

public:
    cpu_ops(int n_threads = 1) : m_ncores(n_threads) {}

    tensor_t add(tensor_t, tensor_t);
    tensor_t sub(tensor_t, tensor_t);
    tensor_t mul(tensor_t, number_t);
    tensor_t div(tensor_t, number_t);

    tensor_t hadamard(tensor_t, tensor_t);
    tensor_t mat_mul(tensor_t, tensor_t);
};

END_BLUST_NAMESPACE