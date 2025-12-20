#pragma once

#include "operations.hpp"
#include <blust/settings.hpp>

START_BLUST_NAMESPACE

class opencl_ops : public operations
{

#if ENABLE_OPENCL_BACKEND

    constexpr static const char* program_file = "opencl-kernel.cl";
    constexpr static const char* VECTOR_ADD_KERNEL_NAME = "vector_add";
    constexpr static const char* HADAMARD_KERNEL_NAME = "vector_hadamard";
    constexpr static const char* MAT_MUL_KERNEL_NAME = "mat_mul_tiled";
    constexpr static const size_t MAT_MUL_TILE_SIZES[2] = {16, 16};
    constexpr static const size_t MAT_MUL_SIZE_FACTOR[2] = {1, 1};

    typedef cl::KernelFunctor<
        cl::Buffer, cl::Buffer, cl::Buffer, unsigned int, float, float> vec_kernel_t;
    typedef std::unique_ptr<vec_kernel_t> vec_kernel_ptr_t;

    typedef cl::KernelFunctor<
        cl::Buffer, cl::Buffer, cl::Buffer, 
        unsigned int, unsigned int, unsigned int> hadamard_kernel_t;
    typedef std::unique_ptr<hadamard_kernel_t> hadamard_kernel_ptr_t;

    typedef cl::KernelFunctor<
        const cl::Buffer, const cl::Buffer, cl::Buffer, 
        const unsigned int, const unsigned int, const unsigned int> mat_mul_kernel_t;
    typedef std::unique_ptr<mat_mul_kernel_t> MAT_MUL_KERNEL_NAME_ptr_t;

    cl::Program m_program;

    cl::Kernel M_get_kernel(const std::string& kernel_name) {
        cl_int err;
        cl::Kernel kernel(m_program, kernel_name.c_str(), &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to create OpenCL kernel: " + kernel_name);
        }
        return kernel;
    }

    vec_kernel_ptr_t m_impl_vec_add;
    hadamard_kernel_ptr_t m_impl_hadamard;
    MAT_MUL_KERNEL_NAME_ptr_t m_impl_mat_mul;

public:
    opencl_ops();

    using ops_tensor_t = operations::ops_tensor_t;

    ops_tensor_t add(tensor_t, tensor_t) override;
    ops_tensor_t sub(tensor_t, tensor_t) override;
    ops_tensor_t mul(tensor_t, number_t) override;
    ops_tensor_t div(tensor_t, number_t) override;

    ops_tensor_t hadamard(tensor_t, tensor_t) override;
    ops_tensor_t mat_mul(tensor_t, tensor_t) override;
    ops_tensor_t transpose(tensor_t) override;
private:
    ops_tensor_t M_perform_vector_like(
        tensor_t a,
        tensor_t b,
        float alpha,
        float beta,
        vec_kernel_t& kernel,
        bool allocate);


#else 
public:
    opencl_ops() = default;

    using ops_tensor_t = operations::ops_tensor_t;

    using operations::add;
    using operations::sub;
    using operations::mul;
    using operations::div;
    using operations::hadamard;
    using operations::mat_mul;
    using operations::transpose;

    // Dummy implementations when CUDA backend is disabled
    void add(ops_tensor_t& a, ops_tensor_t& b, ops_tensor_t& res) override {}
    void sub(ops_tensor_t& a, ops_tensor_t& b, ops_tensor_t& res) override {}
    void mul(ops_tensor_t& a, number_t b, ops_tensor_t& res) override {}
    void div(ops_tensor_t& a, number_t b, ops_tensor_t& res) override {}
    void hadamard(ops_tensor_t& a, ops_tensor_t& b, ops_tensor_t& res) override {}
    void mat_mul(ops_tensor_t& a, ops_tensor_t& b, ops_tensor_t& res) override {}
    void transpose(ops_tensor_t& a, ops_tensor_t& res) override {}
#endif
};

END_BLUST_NAMESPACE