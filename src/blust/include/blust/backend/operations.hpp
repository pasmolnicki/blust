#pragma once

#include <blust/macros.hpp>
#include <blust/base_types.hpp>
#include <blust/tensor.hpp>

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
        
        return ops_tensor(a.layout());
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
    ops_tensor(const tensor& t) { M_borrow(t); }
    ops_tensor(tensor&& t) : tensor(std::forward<tensor>(t)) {}
    ops_tensor& operator=(const tensor& t)
    {
        tensor::operator=(t);
        return *this;
    }

    ops_tensor& operator=(tensor&& t)
    {
        tensor::operator=(std::forward<tensor>(t));
        return *this;
    }

    // Perform a 'smart' copy
    ops_tensor(const ops_tensor& other) 
    {
        M_borrow(other);
        m_in_operation = other.m_in_operation;
    }

    ops_tensor& operator=(const ops_tensor& other)
    {
        if (this != &other)
        {
            m_in_operation = other.m_in_operation;
            tensor::operator=(other);
        }
        return *this;
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

    // Get the released pointer
    ops_tensor(const shape& dim, number_t init = 0.0) : tensor(dim, init) {}
    ops_tensor(tensor::cu_pointer cu_ptr, shape dim) : tensor(cu_ptr, dim) {}

    // Make shared tensor
    ops_tensor(data_handler<number_t>& handler, shape& dim) : tensor(handler, dim) {}

    // Return wheter the tensor is in stacked operation
    inline bool in_operation() const noexcept { return m_in_operation; }

    // Set the flag
    void set_in_operation(bool ops) noexcept { m_in_operation = ops; }
};

class operations
{
public:
    typedef ops_tensor tensor_t;
    typedef ops_tensor tensor_rref_t;
    typedef std::function<tensor_rref_t(tensor_t, tensor_t)> func_t;

    operations() = default;
    virtual ~operations() = default;

    // Any tensor rank operations
    virtual tensor_rref_t sub(tensor_t, tensor_t, bool allocate = true) = 0;
    virtual tensor_rref_t add(tensor_t, tensor_t, bool allocate = true) = 0;
    virtual tensor_rref_t mul(tensor_t, number_t, bool allocate = true) = 0;
    virtual tensor_rref_t div(tensor_t, number_t, bool allocate = true) = 0;

    // 1D operations
    virtual tensor_rref_t hadamard(tensor_t, tensor_t, bool allocate = true) = 0;

    // 2D operations
    virtual tensor_rref_t mat_mul(tensor_t, tensor_t) = 0;
    virtual tensor_rref_t transpose(tensor_t) = 0;

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

