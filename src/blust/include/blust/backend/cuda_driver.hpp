#pragma once

#include <blust/backend/base_backend.hpp>

#include <cuda.h>
#include "helper_cuda.h"
#include "helper_cuda_drvapi.h"

#include <math.h>

START_BLUST_NAMESPACE

// Cuda backend for matrix operations, only 1 backend can be used at a time
class cuda_backend : public base_memory_backend
{
	bool m_available = false;

	CUdevice cuDevice;
	CUcontext cuContext;
	CUmodule cuModule;

	// Element wise operations with 2 vectors
	CUfunction cu_vector_add;
	CUfunction cu_vector_sub;
	CUfunction cu_vector_mul_hadamard;

	// Other operations
	CUfunction cu_vector_mul_scalar;
	CUfunction cu_mat_transpose;
	CUfunction cu_mat_mul;

	size_t m_data1_size = 0;
	size_t m_data2_size = 0;
	size_t m_result_size = 0;

	CUdeviceptr deviceData1 = 0;
	CUdeviceptr deviceData2 = 0;
	CUdeviceptr deviceDataResult = 0;

	void M_run_test();
	void M_prepare_cuda(size_t r, number_t* mat1, size_t m1, number_t* mat2, size_t m2);
	void M_prepare_cuda(size_t r, number_t* mat, size_t m);
	void M_lanuch_vector_like_kernel(number_t* res, number_t* mat1, number_t* mat2, size_t N, CUfunction kernel);

	// Copy the device result to `res`
	inline void M_copy_gpu_result(number_t* res, size_t r)
	{
		checkCudaErrors(cuMemcpyDtoH(res, deviceDataResult, r * sizeof(number_t)));
	}

	// Safe deallocation of a pointer (checks if the pointer is not null)
	static inline void M_safe_dealloc(CUdeviceptr& ptr) 
	{
		if (ptr != 0) {
			checkCudaErrors(cuMemFree(ptr));
			ptr = 0;
		}
	}

	// Try to allocate memory if the previous size is less than the current size
	// And copy the data to the device
	static inline void M_try_alloc(CUdeviceptr& ptr, size_t size, size_t& prev_size) 
	{
		if (prev_size < size) {
			prev_size = size;
			M_safe_dealloc(ptr);
			checkCudaErrors(cuMemAlloc(&ptr, size * sizeof(number_t)));
		}
	}

	// Launch the kernel
	static inline void M_launch_kernel(CUfunction kernel, int nblocks, void** args) {
		checkCudaErrors(cuLaunchKernel(kernel, nblocks, 1, 1,
			THREADS_PER_BLOCK, 1, 1, 0, NULL, args, NULL));
	}

	// Get the number of blocks needed for the given number of elements
	constexpr static int M_get_blocks_per_grid(int N) {
		return std::min<int>((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, MAX_BLOCKS_PER_GRID);
	}

public:
	constexpr static const char* FATBIN_FILE = "cuda_kernel.fatbin";

	constexpr static const char* VECTOR_ADD = "cu_vector_add";
	constexpr static const char* VECTOR_SUB = "cu_vector_sub";
	constexpr static const char* VECTOR_MUL_SCALAR = "cu_vector_mul_scalar";
	constexpr static const char* VECTOR_MUL_HADAMARD = "cu_vector_mul_hadamard";
	constexpr static const char* MAT_TRANSPOSE = "cu_mat_transpose";
	constexpr static const char* MAT_MUL = "cu_mat_mul";

	constexpr static const int THREADS_PER_BLOCK = 256;
	constexpr static const int MAX_BLOCKS_PER_GRID = 32;

	cuda_backend() = default;
	cuda_backend(int argc, char **argv);
	cuda_backend(const cuda_backend& other) = delete;
	cuda_backend(cuda_backend&& other) = delete;
	
	~cuda_backend();

	void init(int argc, char** argv);

	// See if the cuda backend is available
	bool is_available() const { return m_available; }

	// Free the memory (will be called anyway by the destructor)
	void free_memory();

	// Reserve memory for the given size (on 3 buffers)
	// If this is the maximum size of all buffers that will be used
	// then this will be the only memory allocation on the GPU
	void reserve(size_t size_bytes) override;

	// Return the name of the backend 'cuda'
	const char* get_name() override { return "cuda"; }

	// Add `mat1` and `mat2` and store the result in `res` (Rij = M1ij + M2ij)
	void vector_add(number_t* res, number_t* mat1, number_t* mat2, size_t N) override
	{
		M_lanuch_vector_like_kernel(res, mat1, mat2, N, cu_vector_add);
	}

	// Subtract `mat2` from `mat1` and store the result in `res` (Rij = M1ij - M2ij)
	void vector_sub(number_t* res, number_t* mat1, number_t* mat2, size_t N) override
	{
		M_lanuch_vector_like_kernel(res, mat1, mat2, N, cu_vector_sub);
	}

	// Multiply `mat1` and `mat2` element wise and store the result in `res` (Rij = M1ij * M2ij)
	void vector_mul_hadamard(number_t* res, number_t* mat1, number_t* mat2, size_t N) override
	{
		M_lanuch_vector_like_kernel(res, mat1, mat2, N, cu_vector_mul_hadamard);
	}

	// Multiply the matrix by a scalar (Rij = Mij * scalar)
	void vector_scalar_mul(number_t* res, number_t* mat, number_t scalar, size_t N) override;
	void mat_transpose(number_t* res, number_t* mat, size_t rows, size_t cols) override;
	void mat_mul(number_t* res, number_t* mat1, number_t* mat2, size_t rows1, size_t cols2, size_t rows2) override;
};

END_BLUST_NAMESPACE