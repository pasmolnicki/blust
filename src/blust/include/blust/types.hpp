#pragma once

#include "namespaces.hpp"
#include "matrix/matrix.hpp"
#include "tensor.hpp"


START_BLUST_NAMESPACE

typedef float number_t;
typedef tensor tensor_t;
typedef matrix<number_t> matrix_t;
typedef std::vector<number_t> vector_t;
typedef std::vector<tensor_t> batch_t;


#if ENABLE_CUDA_BACKEND
#include <cuda.h>
#else 

#endif

END_BLUST_NAMESPACE