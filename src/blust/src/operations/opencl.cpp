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

const char *opencl_ops::M_getErrorString(cl_int error)
{
switch(error){
    // run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
    }
}

opencl_ops::opencl_ops()
{
    // Check if there's any opencl capabale device
    cl_int err;
    auto defaultPlatform = cl::Platform::getDefault(&err);

    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get default platform: " << M_getErrorString(err) << "\n";
        throw std::runtime_error(std::format("[opencl_ops]: Failed to get default platform: {}", M_getErrorString(err)));
    }

    auto ctx = cl::Context::getDefault(&err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get default context: " << M_getErrorString(err) << "\n";
        throw std::runtime_error(std::format("[opencl_ops]: Failed to get default context: {}", M_getErrorString(err)));
    }

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

    m_impl_gemm = std::make_unique<gemm_kernel_t>(
            M_get_kernel("gemm_3"));
    
    g_settings->opencl_context() = opencl_buffer_context(ctx, queue);
}


typedef operations::ops_tensor_t ops_tensor_t;

void opencl_ops::M_perform_vector_like(
    ops_tensor_t& a, ops_tensor_t& b,
    ops_tensor_t& result,
    float alpha, float beta,
    vec_kernel_t& kernel
) 
{
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
}

/**
 * @brief Add two tensors and return the result
 */
void opencl_ops::add(ops_tensor_t& a, ops_tensor_t& b, ops_tensor_t& res) 
{
    M_perform_vector_like(a, b, res, 1.0, 1.0, *m_impl_vec_add);
}

/**
 * @brief Perform substaction (a - b) and return the result
 */
void opencl_ops::sub(ops_tensor_t& a, ops_tensor_t& b, ops_tensor_t& res) 
{
    M_perform_vector_like(a, b, res, 1.0, -1.0, *m_impl_vec_add);
}

/**
 * @brief Caluculate Ri = Ai * b (see hadamard for element-wise multiplication)
 */
void opencl_ops::mul(ops_tensor_t& a, number_t b, ops_tensor_t& res) 
{
    M_perform_vector_like(a, a, res, b, 0.0, *m_impl_vec_add);
}

/**
 * @brief Calculate Ri = Ai / b
 */
void opencl_ops::div(ops_tensor_t& a, number_t b, ops_tensor_t& res) 
{
    M_perform_vector_like(a, a, res, 1.0 / b, 0.0, *m_impl_vec_add);
}

/**
 * @brief Get the hadamard product: Ci = Ai * Bi
 */
void opencl_ops::hadamard(ops_tensor_t& a, ops_tensor_t& b, ops_tensor_t& res) 
{
    auto queue = g_settings->opencl_context().queue();
    (*m_impl_hadamard)(
        cl::EnqueueArgs(
            queue,
            cl::NDRange(a.size())
        ),
        a.handler().cl_data(),
        b.handler().cl_data(),
        res.handler().cl_data(),
        static_cast<unsigned int>(a.size())
    );
    queue.finish();
}

/**
 * @brief Perform matrix multiplication, and return the result
 * @param a the first matrix, with dimensions m x n and in a column-major order
 * @param b the second matrix, with dimensions n x k and in a column-major order
 * @return the result matrix, with dimensions m x k and in a column-major order
 */
void opencl_ops::mat_mul(ops_tensor_t& a, ops_tensor_t& b, ops_tensor_t& res) {
    // ops_tensor_t result;
    // result.build({m, n}, 0.0f, tensor_t::pointer_type::opencl);
    size_t m = a.dim()[0];
    size_t k = a.dim()[1];
    size_t n = b.dim()[1];

    auto& queue = g_settings->opencl_context().queue();

    // (*m_impl_mat_mul)(
    //     cl::EnqueueArgs(
    //         queue,
    //         cl::NDRange((m + MAT_MUL_SIZE_FACTOR[0] - 1) / MAT_MUL_SIZE_FACTOR[0], (n + MAT_MUL_SIZE_FACTOR[1] - 1) / MAT_MUL_SIZE_FACTOR[1]),
    //         cl::NDRange(MAT_MUL_TILE_SIZES[0], MAT_MUL_TILE_SIZES[1])
    //     ),
    //     a.handler().cl_data(),
    //     b.handler().cl_data(),
    //     res.handler().cl_data(),
    //     static_cast<unsigned int>(m),
    //     static_cast<unsigned int>(k),
    //     static_cast<unsigned int>(n)
    // );

    (*m_impl_gemm)(
        cl::EnqueueArgs(
            queue,
            cl::NDRange(
                utils::get_padded_size(m, GEMM_TILE_SIZES[0]) / GEMM_WORK_PER_THREAD[0],
                utils::get_padded_size(n, GEMM_TILE_SIZES[1]) / GEMM_WORK_PER_THREAD[1]
            ),
            cl::NDRange(
                GEMM_TILE_SIZES[0] / GEMM_WORK_PER_THREAD[0], 
                GEMM_TILE_SIZES[1] / GEMM_WORK_PER_THREAD[1]
            )
        ),
        a.handler().cl_data(),
        b.handler().cl_data(),
        res.handler().cl_data(),
        static_cast<number_t>(1.0),
        static_cast<number_t>(0.0),
        static_cast<unsigned int>(m),
        static_cast<unsigned int>(k),
        static_cast<unsigned int>(n)
    );

    queue.finish();
}

void opencl_ops::transpose(ops_tensor_t& a, ops_tensor_t& res) {
}

END_BLUST_NAMESPACE

#endif // ENABLE_OPENCL_BACKEND