#pragma once

#include "base_backend.hpp"

START_BLUST_NAMESPACE

// CPU backend for matrix operations
class cpu_backend : public base_backend
{
public:
	cpu_backend() = default;

	cpu_backend(const cpu_backend& other) = delete;
	cpu_backend(cpu_backend&& other) = delete;

	// Return the name of the backend 'cpu'
	const char* get_name() override { return "cpu"; }

	void vector_add(number_t* res, number_t* mat1, number_t* mat2, size_t N) override;
	void vector_sub(number_t* res, number_t* mat1, number_t* mat2, size_t N) override;
	void vector_scalar_mul(number_t* res, number_t* mat, number_t scalar, size_t N) override;
	void vector_mul_hadamard(number_t* res, number_t* mat1, number_t* mat2, size_t N) override;
	void mat_transpose(number_t* res, number_t* mat, size_t rows, size_t cols) override;
	void mat_mul(number_t* res, number_t* mat1, number_t* mat2, size_t rows1, size_t cols2, size_t rows2) override;
};

END_BLUST_NAMESPACE