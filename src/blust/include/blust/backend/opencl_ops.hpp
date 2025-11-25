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
    constexpr static const size_t MAT_MUL_TILE_SIZE = 16;

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

    using tensor_rref_t = operations::tensor_rref_t;

    tensor_rref_t add(tensor_t, tensor_t, bool allocate = true) override;
    tensor_rref_t sub(tensor_t, tensor_t, bool allocate = true) override;
    tensor_rref_t mul(tensor_t, number_t, bool allocate = true) override;
    tensor_rref_t div(tensor_t, number_t, bool allocate = true) override;

    tensor_rref_t hadamard(tensor_t, tensor_t, bool allocate = true) override;
    tensor_rref_t mat_mul(tensor_t, tensor_t) override;
    tensor_rref_t transpose(tensor_t) override;
private:
    tensor_rref_t M_perform_vector_like(
        tensor_t a,
        tensor_t b,
        float alpha,
        float beta,
        vec_kernel_t& kernel,
        bool allocate);


#else 
public:
    opencl_ops() = default;

    // Dummy implementations when CUDA backend is disabled
    tensor_rref_t add(tensor_t a, tensor_t b, bool allocate = true) override { return a; }
    tensor_rref_t sub(tensor_t a, tensor_t b, bool allocate = true) override { return a; }
    tensor_rref_t mul(tensor_t a, number_t b, bool allocate = true) override { return a; }
    tensor_rref_t div(tensor_t a, number_t b, bool allocate = true) override { return a; }
    tensor_rref_t hadamard(tensor_t a, tensor_t b, bool allocate = true) override { return a; }
    tensor_rref_t mat_mul(tensor_t a, tensor_t b) override { return a; }
    tensor_rref_t transpose(tensor_t a) override { return a; }
#endif
};

END_BLUST_NAMESPACE