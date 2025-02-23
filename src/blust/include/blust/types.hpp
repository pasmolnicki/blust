#pragma once

#include "namespaces.hpp"
#include "matrix/matrix.hpp"


START_BLUST_NAMESPACE

typedef float number_t;
typedef matrix<number_t> matrix_t;
typedef std::vector<number_t> vector_t;
typedef std::vector<matrix_t> batch_t;

END_BLUST_NAMESPACE