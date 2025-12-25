#pragma once

#include <blust/macros.hpp>
#include <blust/base_types.hpp>
#include <blust/tensor.hpp>
#include <blust/types.hpp>

#include "ops_tensor.hpp"
#include "nn_ops.hpp"

START_BLUST_NAMESPACE

class operations : public nn_ops
{
public:
    typedef std::function<tensor_t(tensor_t, tensor_t)> func_t;

    operations() = default;
    virtual ~operations() = default;

    using nn_ops::nn_hidden_gradient;
    using nn_ops::nn_output_gradient;
    using nn_ops::nn_feed_forward;

    // Any tensor rank operations
    virtual void sub(tensor_t&, tensor_t&, tensor_t&) = 0;
    virtual void add(tensor_t&, tensor_t&, tensor_t&) = 0;
    virtual void mul(tensor_t&, number_t, tensor_t&) = 0;
    virtual void div(tensor_t&, number_t, tensor_t&) = 0;
    virtual void hadamard(tensor_t&, tensor_t&, tensor_t&) = 0;
    virtual void mat_mul(tensor_t&, tensor_t&, tensor_t&) = 0;
    virtual void transpose(tensor_t&, tensor_t&) = 0;


    // 1D operations
    ops_tensor_t sub(ops_tensor_t a, ops_tensor_t b) {
        ops_tensor res = ops_tensor::try_borrow(a, b);
        sub(a, b, res);
        return std::move(res);
    }

    ops_tensor_t add(ops_tensor_t a, ops_tensor_t b) {
        ops_tensor res = ops_tensor::try_borrow(a, b);
        add(a, b, res);
        return std::move(res);
    }

    ops_tensor_t hadamard(ops_tensor_t a, ops_tensor_t b) {
        ops_tensor res = ops_tensor::try_borrow(a, b);
        hadamard(a, b, res);
        return std::move(res);
    }

    // Scalar operations
    ops_tensor_t mul(ops_tensor_t a, number_t b) {
        ops_tensor res = ops_tensor::try_borrow(a);
        mul(a, b, res);
        return std::move(res);
    }

    ops_tensor_t div(ops_tensor_t a, number_t b) {
        ops_tensor res = ops_tensor::try_borrow(a);
        div(a, b, res);
        return std::move(res);
    }

    // 2D operations
    virtual ops_tensor_t mat_mul(ops_tensor_t a, ops_tensor_t b) {
        M_assert_tensor_dim_mat_mul(a, b);
        ops_tensor res({ a.dim()[0], b.dim()[1] }, 0.0, a.type());
        mat_mul(a, b, res);
        return std::move(res);
    }

    ops_tensor_t transpose(ops_tensor_t a) {
        ops_tensor res({ a.dim()[1], a.dim()[0] }, 0.0, a.type());
        transpose(a, res);
        return std::move(res);
    }

protected:

    static void M_assert_tensor_dim_mat_mul(tensor_t& a, tensor_t& b) noexcept(false)
    {
        if ((a.rank() != 2 || b.rank() != 2) || (a.dim()[1] != b.dim()[0]))
            throw std::runtime_error("Invalid tensor dimensions for matrix multiplication");
    }

    static void M_assert_tensor_same_size(tensor_t& a, tensor_t& b) noexcept(false)
    {
        if ((a.rank() != b.rank()) || (a.size() != b.size()))
            throw std::runtime_error("The tensor's size doesn't match");
    }
};

// Global operation backend, initialized in 'blust.hpp' by init function
extern std::unique_ptr<operations> ops;

END_BLUST_NAMESPACE

