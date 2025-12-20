#pragma once

#include "operations.hpp"

#if ENABLE_CUDA_BACKEND
#   include <cuda.h>
#   include <cuda_runtime_api.h>
#   include <cublas.h>
#endif

START_BLUST_NAMESPACE

class cuda_ops : public operations
{
public:

#if ENABLE_CUDA_BACKEND
    ops_tensor_t add(tensor_t, tensor_t) override;
    ops_tensor_t sub(tensor_t, tensor_t) override;
    ops_tensor_t mul(tensor_t, number_t) override;
    ops_tensor_t div(tensor_t, number_t) override;

    ops_tensor_t hadamard(tensor_t, tensor_t) override;
    ops_tensor_t mat_mul(tensor_t, tensor_t) override;
    ops_tensor_t transpose(tensor_t) override;
#else 
    // Dummy implementations when CUDA backend is disabled
    void add(tensor_t& a, tensor_t& b, tensor_t&) override {}
    void sub(tensor_t& a, tensor_t& b, tensor_t&) override {}
    void mul(tensor_t& a, number_t b, tensor_t&) override {}
    void div(tensor_t& a, number_t b, tensor_t&) override {}
    void hadamard(tensor_t& a, tensor_t& b, tensor_t&) override {}
    void mat_mul(tensor_t& a, tensor_t& b, tensor_t&) override {}
    void transpose(tensor_t& a, tensor_t&) override {}
#endif
};

END_BLUST_NAMESPACE