#include <blust/backend/cpu.hpp>

#include <cmath>

START_BLUST_NAMESPACE

constexpr void d_sigmoid(const_npointer_t output, npointer_t result, size_t n) {
	for (size_t i = 0; i < n; i++) {
		result[i] = output[i] * (1 - output[i]);
	}
}

constexpr void d_relu(const_npointer_t output, npointer_t result, size_t n) {
	for (size_t i = 0; i < n; i++) {
		result[i] = output[i] > 0;
	}
}

constexpr void d_softmax(const_npointer_t output, npointer_t result, size_t n) {
	for (size_t i = 0; i < n; i++) {
		result[i] = output[i] * (1 - output[i]);
	}
}

constexpr void d_none(const_npointer_t, npointer_t result, size_t n) {
	memset(result, 1, n * sizeof(number_t));
}


npointer_t cpu_backend::M_get_d_activations(const_npointer_t outputs, size_t n, activations act_type) {
	const auto result = new number_t[n];

	switch (act_type) {
		case activations::sigmoid:
			d_sigmoid(outputs, result, n);
			break;
		case activations::softmax:
			d_softmax(outputs, result, n);
			break;
		case activations::relu:
			d_relu(outputs, result, n);
			break;
		case activations::none:
		default:
			d_none(outputs, result, n);
			break;
	}

	return result;
}


void cpu_backend::relu(npointer_t input, npointer_t result, size_t n) {
	for (size_t i = 0; i < n; i++) {
		result[i] = input[i] > 0 ? input[i] : 0;
	}
}

void cpu_backend::sigmoid(npointer_t input, npointer_t result, size_t n) {
	for (size_t i = 0; i < n; i++) {
		result[i] = (number_t)1.0 / ((number_t)1.0 + (number_t)std::exp(-input[i]));
	}
}

void cpu_backend::softmax(npointer_t input, npointer_t result, size_t n) {
	number_t sum = 0;
	for (size_t i = 0; i < n; i++) {
		result[i] = std::exp(input[i]);
		sum += result[i];
	}

	sum += 1e-4;

	for (size_t i = 0; i < n; i++) {
		result[i] = result[i] / sum;
	}
}

void cpu_backend::backprop_dense_output(
	number_t *outputs, number_t *expected, activations act_type,
	number_t *parial_deriv, shape2D output_shape, size_t n_batch)
{
	for (size_t i = 0; i < output_shape.x; i++) {


		for (size_t j = 0; j < output_shape.y; j++) {}
	}
}


void cpu_backend::backprop_hidden_dense(
	number_t *d_weights, number_t *d_biases, activations act_type, number_t *d_prev_activations,
	number_t *weights, number_t *inputs, number_t *prev_d_activations, number_t *prev_weights,
	size_t n_weights, size_t n_prev_activations, size_t n_inputs, size_t n_batch)
{

}

// Add the matrices (Cij = Aij + Bij)
void cpu_backend::vector_add(number_t* res, number_t* mat1, number_t* mat2, size_t N)
{
	for (size_t i = 0; i < N; i++)
		res[i] = mat1[i] + mat2[i];
}

// Subtract the matrices (Cij = Aij - Bij)
void cpu_backend::vector_sub(number_t* res, number_t* mat1, number_t* mat2, size_t N)
{
	for (size_t i = 0; i < N; i++)
		res[i] = mat1[i] - mat2[i];
}

// Multiply the matrix by a scalar (k * M)
void cpu_backend::vector_scalar_mul(number_t* res, number_t* mat1, number_t scalar, size_t N)
{
	for (size_t i = 0; i < N; i++)
		res[i] = mat1[i] * scalar;
}

// Hadamard product, element-wise multiplication (Cij = Aij * Bij)
void cpu_backend::vector_mul_hadamard(number_t* res, number_t* mat1, number_t* mat2, size_t N)
{
	for (size_t i = 0; i < N; i++)
		res[i] = mat1[i] * mat2[i];
}

// Transpose the matrix
void cpu_backend::mat_transpose(number_t* res, number_t* mat, size_t rows, size_t cols)
{
	size_t size = rows * cols;
	for (size_t n = 0; n < size; ++n)
	{
		size_t i = n / rows;
		size_t j = n % rows;
		res[n] = mat[cols * j + i];
	}
}

// Typical matrix multiplication
void cpu_backend::mat_mul(number_t* res, number_t* mat1, number_t* mat2, size_t rows1, size_t cols2, size_t rows2)
{
	cpu_ops::M_impl_matumul<cpu_ops::matmul_type::see>(
		mat1, rows2, mat2, cols2, res, cols2, rows1, cols2, rows2
	);
	// for (size_t r1 = 0; r1 < rows1; r1++) 
	// {
	// 	for (size_t k = 0; k < cols2; ++k)
	// 	{
	// 		number_t sum = 0;
	// 		for (size_t c1 = 0; c1 < rows2; c1++)
	// 			sum += mat1[r1 * rows2 + c1] * mat2[c1 * cols2 + k]; 

	// 		res[r1 * cols2 + k] = sum;
	// 	}
	// }
}

END_BLUST_NAMESPACE