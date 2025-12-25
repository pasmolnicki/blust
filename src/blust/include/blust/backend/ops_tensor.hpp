#pragma once

#include <blust/macros.hpp>
#include <blust/base_types.hpp>
#include <blust/tensor.hpp>
#include <blust/types.hpp>


START_BLUST_NAMESPACE

// I can not use the `tensor` here, If I want optimized perfomance on cuda and cpu
// Because I don't know wheter I am in an `operation` or I am just copying the tensor
// So I should make a indirect object, a middle man, with possible conversion to tensor,
// and be able to be constructed from a tensor (when user passes the tensor as an argument)
// Hence I am creating the `ops_tensor`, which will not copy the buffer from gpu device or heap memory
// thus optimizing the perfomance (tensor will always copy the buffer from the gpu)

// Helper class for proper memory management inside chained operations
class ops_tensor : public tensor
{
    bool m_in_operation = false;

public:
    // Get the tensor based on the 'operation' flag, if a or b is 'in operation'
    // then borrow their buffer, else create tensor with newly allocated memory
    static inline ops_tensor try_borrow(ops_tensor& a, ops_tensor& b) noexcept
    {
        // If a or b is in operation, create a shared result buffer
        if (a.in_operation()) 
            return ops_tensor(a.m_handler, a.m_shape);

        if (b.in_operation()) 
            return ops_tensor(b.m_handler, b.m_shape);
        
        return ops_tensor(a.layout(), 0.0, a.type());
    }

    static inline ops_tensor try_borrow(ops_tensor& a) noexcept
    {
        if (a.in_operation()) 
            return ops_tensor(a.m_handler, a.m_shape);
        
        return ops_tensor(a.layout());
    }



    friend class cpu_ops;

    ops_tensor() = default;

    // Borrow the buffer
    ops_tensor(const tensor& t) { *this = t; }
    ops_tensor(tensor&& t) : tensor(std::forward<tensor>(t)) {}
    ops_tensor& operator=(const tensor& t)
    {
        M_borrow(t);
        return *this;
    }

    ops_tensor& operator=(tensor&& t)
    {
        tensor::operator=(std::forward<tensor>(t));
        return *this;
    }

    // Perform a 'smart' copy
    ops_tensor(const ops_tensor& other) {
        void(*this = other);
    }

    ops_tensor& operator=(const ops_tensor& other)
    {
        if (this == &other)
            return *this;

        M_borrow(other);
        m_in_operation = other.m_in_operation;
        return *this;
    }

    // Deep copy the buffer
    void copy(const ops_tensor& other) {
        if (this == &other)
            return;

        // Deep copy the buffer
        tensor::operator=(other);
    }

    // Move constructor
    ops_tensor(ops_tensor&& other) : tensor(std::move(other)), m_in_operation(other.m_in_operation) {}
    ops_tensor& operator=(ops_tensor&& other)
    {
        if (this != &other)
        {
            tensor::operator=(std::move(other));
        }
        return *this;
    }

    // Create new tensor
    ops_tensor(const shape& dim, number_t init = 0.0) : tensor(dim, init) {}
    ops_tensor(const shape& dim, number_t init, pointer_type type) : tensor(dim, init, type) {}

    // Make shared tensor
    ops_tensor(data_handler<number_t>& handler, shape& dim) : tensor(handler, dim), m_in_operation(true) {}

    // Return wheter the tensor is in stacked operation
    inline bool in_operation() const noexcept { return m_in_operation; }

    // Set the flag
    void set_in_operation(bool ops) noexcept { m_in_operation = ops; }
};

using ops_tensor_t = ops_tensor;

END_BLUST_NAMESPACE