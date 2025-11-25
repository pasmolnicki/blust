#include <blust/backend/opencl_ops.hpp>

#include <fstream>
#include <stdexcept>
#include <filesystem>

#if ENABLE_OPENCL_BACKEND

START_BLUST_NAMESPACE

static std::string load_kernel_source(const std::string& filename) {
    std::ifstream file_stream(filename);
    if (!file_stream.is_open()) {
        throw std::runtime_error("Failed to open OpenCL kernel source file: " + filename);
    }
    return std::string((std::istreambuf_iterator<char>(file_stream)),
                       std::istreambuf_iterator<char>());
}

opencl_ops::opencl_ops()
{
    auto ctx = cl::Context(CL_DEVICE_TYPE_DEFAULT);
    auto queue = cl::CommandQueue(ctx);
    m_program = cl::Program(
        ctx, 
        load_kernel_source((g_settings->path() / program_file).string()),
        true
    );

    // Initialize kernels
    m_impl_vec_add = std::make_unique<vec_kernel_t>(
            M_get_kernel(VECTOR_ADD_KERNEL_NAME));
    
    m_impl_hadamard = std::make_unique<hadamard_kernel_t>(
            M_get_kernel(HADAMARD_KERNEL_NAME));

    m_impl_mat_mul = std::make_unique<mat_mul_kernel_t>(
            M_get_kernel(MAT_MUL_KERNEL_NAME));
    
    g_settings->opencl_context() = opencl_buffer_context(ctx, queue);
}


typedef operations::tensor_rref_t tensor_rref_t;

tensor_rref_t opencl_ops::M_perform_vector_like(
    tensor_t a, tensor_t b,
    float alpha, float beta,
    vec_kernel_t& kernel,
    bool allocate
) 
{
    M_assert_tensor_same_size(a, b);

    tensor_rref_t result;
    if (allocate) {
        result.build(a.layout(), 0.0f, tensor_t::pointer_type::opencl);
    } else {
        result = ops_tensor::try_borrow(a, b);
    }

    auto& queue = g_settings->opencl_context().queue();

    // Execute the kernel
    auto err = kernel(
        cl::EnqueueArgs(
            queue,
            cl::NDRange(a.size())
        ),
        a.handler().cl_data(),
        b.handler().cl_data(),
        result.handler().cl_data(),
        static_cast<unsigned int>(a.size()),
        alpha,
        beta
    );

    // if (err.getInfo<CL_QUEUE_STATUS>() != CL_SUCCESS) {
    //     throw std::runtime_error("Failed to enqueue OpenCL kernel for vector operation");
    // }

    queue.finish();

    return result;
}

/**
 * @brief Add two tensors and return the result
 */
tensor_rref_t opencl_ops::add(tensor_t a, tensor_t b, bool allocate) 
{
    return M_perform_vector_like(a, b, 1.0, 1.0, *m_impl_vec_add, allocate);
    // return a;
    // return M_perform_vector_like(a, b, 1.0, 1.0, M_impl_add, allocate);
}

/**
 * @brief Perform substaction (a - b) and return the result
 */
tensor_rref_t opencl_ops::sub(tensor_t a, tensor_t b, bool allocate) 
{
    return M_perform_vector_like(a, b, 1.0, -1.0, *m_impl_vec_add, allocate);
    // return M_perform_vector_like(a, b, 1.0, -1.0, M_impl_add, allocate);
}

/**
 * @brief Caluculate Ri = Ai * b (see hadamard for element-wise multiplication)
 */
tensor_rref_t opencl_ops::mul(tensor_t a, number_t b, bool allocate) 
{
    return std::move(M_perform_vector_like(a, a, b, 0.0, *m_impl_vec_add, allocate));
    // return M_perform_vector_like(a, a, b, 0.0, M_impl_add, allocate); // c = a * b + a * 0
}

/**
 * @brief Calculate Ri = Ai / b
 */
tensor_rref_t opencl_ops::div(tensor_t a, number_t b, bool allocate) {
    return std::move(M_perform_vector_like(a, a, 1.0f / b, 0.0f, *m_impl_vec_add, allocate));
    // return a;
}

/**
 * @brief Get the hadamard product: Ci = Ai * Bi
 */
tensor_rref_t opencl_ops::hadamard(tensor_t a, tensor_t b, bool allocate) {
    return a;
}

/**
 * @brief Perform matrix multiplication, and return the result
 * @param a the first matrix, with dimensions m x n and in a column-major order
 * @param b the second matrix, with dimensions n x k and in a column-major order
 * @return the result matrix, with dimensions m x k and in a column-major order
 */
tensor_rref_t opencl_ops::mat_mul(tensor_t a, tensor_t b) {
    M_assert_tensor_dim_mat_mul(a, b);

    auto m = a.dim()[0],
         k = a.dim()[1],
         n = b.dim()[1];

    tensor_rref_t result;
    result.build({m, n}, 0.0f, tensor_t::pointer_type::opencl);

    auto& queue = g_settings->opencl_context().queue();

    (*m_impl_mat_mul)(
        cl::EnqueueArgs(
            queue,
            cl::NDRange(m, n),
            cl::NDRange(MAT_MUL_TILE_SIZE, MAT_MUL_TILE_SIZE)
        ),
        a.handler().cl_data(),
        b.handler().cl_data(),
        result.handler().cl_data(),
        static_cast<unsigned int>(m),
        static_cast<unsigned int>(k),
        static_cast<unsigned int>(n)
    );

    queue.finish();
    return result;
}

tensor_rref_t opencl_ops::transpose(tensor_t a) {
    return a;
}

END_BLUST_NAMESPACE

#endif // ENABLE_OPENCL_BACKEND