#pragma once

#include <format>
#include <memory>
#include <stdexcept>

#include "operations.hpp"
#include <blust/settings.hpp>

START_BLUST_NAMESPACE

class opencl_ops : public operations
{
public:
    using ops_tensor_t = operations::ops_tensor_t;

    using operations::add;
    using operations::sub;
    using operations::mul;
    using operations::div;
    using operations::hadamard;
    using operations::mat_mul;
    using operations::transpose;

#if ENABLE_OPENCL_BACKEND
private:
    constexpr static const char* program_file = "opencl-kernel.cl";
    constexpr static const char* VECTOR_ADD_KERNEL_NAME = "vector_add";
    constexpr static const char* HADAMARD_KERNEL_NAME = "vector_hadamard";
    constexpr static const char* MAT_MUL_KERNEL_NAME = "mat_mul_tiled";
    constexpr static const size_t GEMM_TILE_SIZES[2] = {32, 32}; // work-group size
    constexpr static const size_t GEMM_WORK_PER_THREAD[2] = {1, 8}; // each work-item computes 1x1 elements

    typedef cl::KernelFunctor<
        cl::Buffer, cl::Buffer, cl::Buffer, unsigned int, float, float> vec_kernel_t;
    typedef std::unique_ptr<vec_kernel_t> vec_kernel_ptr_t;

    typedef cl::KernelFunctor<
        cl::Buffer, cl::Buffer, cl::Buffer, unsigned int> hadamard_kernel_t;
    typedef std::unique_ptr<hadamard_kernel_t> hadamard_kernel_ptr_t;

    typedef cl::KernelFunctor<
        const cl::Buffer, const cl::Buffer, cl::Buffer, 
        const unsigned int, const unsigned int, const unsigned int> mat_mul_kernel_t;
    typedef std::unique_ptr<mat_mul_kernel_t> mat_mul_kernel_ptr_t;

    
    typedef cl::KernelFunctor<
        cl::Buffer, cl::Buffer, cl::Buffer, 
        number_t, number_t, unsigned int, unsigned int, unsigned int> gemm_kernel_t;
    typedef std::unique_ptr<gemm_kernel_t> gemm_kernel_ptr_t;

    cl::Program m_program;

    const char* M_getErrorString(cl_int error);

    cl::Kernel M_get_kernel(const std::string& kernel_name) {
        cl_int err;
        cl::Kernel kernel(m_program, kernel_name.c_str(), &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error(std::format("Failed to create OpenCL kernel: {} (err:{})", kernel_name, M_getErrorString(err)));
        }
        return kernel;
    }

    void M_perform_vector_like(
        ops_tensor_t& a,
        ops_tensor_t& b,
        ops_tensor_t& res,
        float alpha,
        float beta,
        vec_kernel_t& kernel);

    gemm_kernel_ptr_t m_impl_gemm;
    vec_kernel_ptr_t m_impl_vec_add;
    hadamard_kernel_ptr_t m_impl_hadamard;
    mat_mul_kernel_ptr_t m_impl_mat_mul;

public:
    opencl_ops();

    // Dummy implementations when CUDA backend is disabled
    void add(ops_tensor_t& a, ops_tensor_t& b, ops_tensor_t& res) override;
    void sub(ops_tensor_t& a, ops_tensor_t& b, ops_tensor_t& res) override;
    void mul(ops_tensor_t& a, number_t b, ops_tensor_t& res) override;
    void div(ops_tensor_t& a, number_t b, ops_tensor_t& res) override;
    void hadamard(ops_tensor_t& a, ops_tensor_t& b, ops_tensor_t& res) override;
    void mat_mul(ops_tensor_t& a, ops_tensor_t& b, ops_tensor_t& res) override;
    void transpose(ops_tensor_t& a, ops_tensor_t& res) override;  

#else 
    opencl_ops() = default;

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