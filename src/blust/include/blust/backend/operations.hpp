#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <blust/base_types.hpp>
#include <blust/tensor.hpp>

START_BLUST_NAMESPACE

// I can not use the `tensor` here, If I want optimized perfomance on cuda
// Because I don't know wheter I am in an `operation` or I am just copying the tensor
// So I should make a indirect object, a middle man, with possible conversion to tensor,
// and be able to be constructed from a tensor (when user passes the tensor as an argument)
// Hence I am creating the `ops_tensor`, which will not copy the buffer from gpu device 
// thus optimizing the perfomance (tensor will always copy the buffer from the gpu)


class ops_tensor : public tensor
{
public:

    // Get the ops tensor from a tensor
    static ops_tensor get(const tensor& t)
    {
        if (t.is_cuda())
            return ops_tensor(t.cu_release(), t.layout());
        return ops_tensor(t.release(), t.layout()); // idk if i should do that
    }

    // Release the pointer of the t
    ops_tensor(const tensor& t)
    {

    }

    ops_tensor(ops_tensor&& t)
    {

    }

    // Get the released pointer
    ops_tensor(tensor::cu_pointer cu_ptr, shape dim) : tensor(cu_ptr, dim) {}
    ops_tensor(tensor::pointer data, shape dim) : tensor(data, dim) {}
};

class operations
{
public:
    typedef ops_tensor tensor_t;

    operations() = default;
    virtual ~operations() = default;

    // Any tensor rank operations
    virtual tensor_t sub(tensor_t, tensor_t) = 0;
    virtual tensor_t add(tensor_t, tensor_t) = 0;
    virtual tensor_t mul(tensor_t, number_t) = 0;
    virtual tensor_t div(tensor_t, number_t) = 0;

    // 1D operations
    virtual tensor_t hadamard(tensor_t, tensor_t) = 0;

    // 2D operations
    virtual tensor_t mat_mul(tensor_t, tensor_t) = 0;

protected:

    void M_assert_tensor_dim_mat_mul(tensor_t& a, tensor_t& b)
    {
        if ((a.rank() != 2 || b.rank() != 2) || (a.dim()[1] != b.dim()[1]))
            throw std::runtime_error("Invalid tensor dimensions for matrix multiplication");
    }

    void M_assert_tensor_same_size(tensor_t& a, tensor_t& b)
    {
        if ((a.rank() != b.rank()) || (a.size() != b.size()))
            throw std::runtime_error("The tensor's size doesn't match");
    }
};

END_BLUST_NAMESPACE

