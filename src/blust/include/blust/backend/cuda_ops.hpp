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
    void add(ops_tensor_t& a, ops_tensor_t& b, ops_tensor_t&) override {}
    void sub(ops_tensor_t& a, ops_tensor_t& b, ops_tensor_t&) override {}
    void mul(ops_tensor_t& a, number_t b, ops_tensor_t&) override {}
    void div(ops_tensor_t& a, number_t b, ops_tensor_t&) override {}
    void hadamard(ops_tensor_t& a, ops_tensor_t& b, ops_tensor_t&) override {}
    void mat_mul(ops_tensor_t& a, ops_tensor_t& b, ops_tensor_t&) override {}
    void transpose(ops_tensor_t& a, ops_tensor_t&) override {}
#endif
};

END_BLUST_NAMESPACE