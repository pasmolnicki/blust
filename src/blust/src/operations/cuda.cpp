#include <blust/backend/cuda_ops.hpp>


#if ENABLE_CUDA_BACKEND

START_BLUST_NAMESPACE

typedef operations::ops_tensor_t ops_tensor_t;

/**
 * @brief Add two tensors and return the result
 */
ops_tensor_t cuda_ops::add(tensor_t a, tensor_t b) 
{
    return a;
    // return M_perform_vector_like(a, b, 1.0, 1.0, M_impl_add, allocate);
}

/**
 * @brief Perform substaction (a - b) and return the result
 */
ops_tensor_t cuda_ops::sub(tensor_t a, tensor_t b) 
{
    return a;
    // return M_perform_vector_like(a, b, 1.0, -1.0, M_impl_add, allocate);
}

/**
 * @brief Caluculate Ri = Ai * b (see hadamard for element-wise multiplication)
 */
ops_tensor_t cuda_ops::mul(tensor_t a, number_t b) 
{
    return a;
    // return M_perform_vector_like(a, a, b, 0.0, M_impl_add, allocate); // c = a * b + a * 0
}

/**
 * @brief Calculate Ri = Ai / b
 */
ops_tensor_t cuda_ops::div(tensor_t a, number_t b) {
    return a;
}

/**
 * @brief Get the hadamard product: Ci = Ai * Bi
 */
ops_tensor_t cuda_ops::hadamard(tensor_t a, tensor_t b) { return a; }

/**
 * @brief Perform matrix multiplication, and return the result
 * @param a the first matrix, with dimensions m x n and in a column-major order
 * @param b the second matrix, with dimensions n x k and in a column-major order
 * @return the result matrix, with dimensions m x k and in a column-major order
 */
ops_tensor_t cuda_ops::mat_mul(tensor_t a, tensor_t b) {
    return a;
}

END_BLUST_NAMESPACE

#endif // ENABLE_CUDA_BACKEND