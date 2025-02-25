#pragma once

#include <blust/base_types.hpp>

#include <memory>

START_BLUST_NAMESPACE

// Base backend for matrix operations (either cpu or gpu) (only 1 backend can be used at a time)
class base_backend
{
public:
	// Default constructor
	base_backend() = default;
	virtual ~base_backend() = default;

	// Return the name of the backend
	virtual const char* get_name() = 0;

	virtual void vector_add(number_t* res, number_t* mat1, number_t* mat2, size_t N) = 0;
	virtual void vector_sub(number_t* res, number_t* mat1, number_t* mat2, size_t N) = 0;
	virtual void vector_mul_hadamard(number_t* res, number_t* mat1, number_t* mat2, size_t N) = 0;
	virtual void vector_scalar_mul(number_t* res, number_t* mat, number_t scalar, size_t N) = 0;
	virtual void mat_transpose(number_t* res, number_t* mat, size_t rows, size_t cols) = 0;
	virtual void mat_mul(number_t* res, number_t* mat1, number_t* mat2, size_t rows1, size_t cols2, size_t rows2) = 0;
};

// The backend that is used for all operations (intialized in main.cpp)
static std::unique_ptr<base_backend> g_backend;
constexpr std::unique_ptr<base_backend>& get_backend() { return g_backend; }

END_BLUST_NAMESPACE