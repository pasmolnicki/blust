#pragma once

#include <chrono>
#include <functional>

#include <blust/utils.hpp>
#include "cuda_driver.hpp"
#include "cpu.hpp"

START_BLUST_NAMESPACE

// Helper class for measuring the execution time of the operation
class Gpu_timer
{
	CUevent m_start;
	CUevent m_stop;
	float m_milliseconds;
public:
	Gpu_timer() {
		cuEventCreate(&m_start, 0);
		cuEventCreate(&m_stop, 0);
		m_milliseconds = 0.0f;
	}
	~Gpu_timer() {
		cuEventDestroy(m_start);
		cuEventDestroy(m_stop);
	}
	void start() { cuEventRecord(m_start, 0); }
	void stop() {
		cuEventRecord(m_stop, 0);
		cuEventSynchronize(m_stop);
		cuEventElapsedTime(&m_milliseconds, m_start, m_stop);
	}
	float get_time() { return m_milliseconds; }
};

// This is optimized backend, it will use the fastest backend available
// Boils down to comparing the execution time of the operation on CPU and GPU
// with given matrix dimension, and choosing the fastest one
class optimized_backend : public base_backend
{
	typedef std::function<void(base_backend*, number_t*, number_t*, number_t*, size_t)> fn_backend_t;

	cpu_backend m_cpu;
	cuda_backend m_cuda;
	size_t m_threshold_vector_size;
	size_t m_threshold_matrix_size;

	void M_set_threshold();

	size_t M_get_size_threshold(fn_backend_t fn_backend);

	void M_set_vector_treshold();
	void M_set_matrix_treshold();

	// Check if the GPU should be used for the given size
	bool M_use_gpu(size_t size, size_t threshold)
	{
		return size > threshold && m_cuda.is_available();
	}
public:

	optimized_backend(int argc, char** argv) : m_cuda(argc, argv)
	{
		M_set_threshold();
	}

	const char* get_name() override { return "optimized"; }


	void vector_add(number_t* res, number_t* mat1, number_t* mat2, size_t N) override
	{
		if (M_use_gpu(N, m_threshold_vector_size))
			m_cuda.vector_add(res, mat1, mat2, N);
		else
			m_cpu.vector_add(res, mat1, mat2, N);
	}

	void vector_sub(number_t* res, number_t* mat1, number_t* mat2, size_t N) override
	{
		if (M_use_gpu(N, m_threshold_vector_size))
			m_cuda.vector_sub(res, mat1, mat2, N);
		else
			m_cpu.vector_sub(res, mat1, mat2, N);
	}
	void vector_scalar_mul(number_t* res, number_t* mat, number_t scalar, size_t N) override
	{
		if (M_use_gpu(N, m_threshold_vector_size))
			m_cuda.vector_scalar_mul(res, mat, scalar, N);
		else
			m_cpu.vector_scalar_mul(res, mat, scalar, N);
	}

	void vector_mul_hadamard(number_t* res, number_t* mat1, number_t* mat2, size_t N) override
	{
		if (M_use_gpu(N, m_threshold_vector_size))
			m_cuda.vector_mul_hadamard(res, mat1, mat2, N);
		else
			m_cpu.vector_mul_hadamard(res, mat1, mat2, N);
	}

	void mat_transpose(number_t* res, number_t* mat, size_t rows, size_t cols) override
	{
		if (M_use_gpu(rows * cols, m_threshold_vector_size))
			m_cuda.mat_transpose(res, mat, rows, cols);
		else
			m_cpu.mat_transpose(res, mat, rows, cols);
	}

	void mat_mul(number_t* res, number_t* mat1, number_t* mat2, size_t rows1, size_t cols2, size_t rows2) override
	{
		if (M_use_gpu(rows2 * cols2,  m_threshold_matrix_size))
			m_cuda.mat_mul(res, mat1, mat2, rows1, cols2, rows2);
		else
			m_cpu.mat_mul(res, mat1, mat2, rows1, cols2, rows2);
	}	
};

END_BLUST_NAMESPACE