#pragma once

#include <blust/types.hpp>

#include <cuda.h>
#include "helper_cuda.h"
#include "helper_cuda_drvapi.h"


START_BLUST_NAMESPACE

enum cuda_function_type {
	vector_add = 0,
	vector_sub
};

void cuda_init(int argc, char** argv);

template <cuda_function_type type>
void lanuch_kernel(matrix_t& res, matrix_t& mat1, matrix_t& mat2);


END_BLUST_NAMESPACE