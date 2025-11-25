#pragma once

#include "namespaces.hpp"

#include <vector>
#include <ostream>

#if ENABLE_CUDA_BACKEND
#   include <cuda.h>
#   include <cuda_runtime_api.h>
#endif

#if ENABLE_OPENCL_BACKEND
#   define CL_TARGET_OPENCL_VERSION 200
#   define CL_HPP_TARGET_OPENCL_VERSION 200
#   include <CL/opencl.hpp>
#endif

START_BLUST_NAMESPACE

typedef float number_t;
typedef number_t* npointer_t;
typedef const number_t* const_npointer_t;
typedef std::vector<number_t> vector_t;

#if !defined(ENABLE_CUDA_BACKEND) || (ENABLE_CUDA_BACKEND == 0)
    typedef unsigned int CUdeviceptr;
#endif

/**
 * ## Activation types
 * - `none`: f(x) = x
 * - `relu`: f(x) = {x if x >= 0, 0 if x < 0}
 * - `sigmoid`: f(x) = 1 / (1 + exp(-x))
 * - `softmax`: f([x1, x2, ..., xn], i) = exp(xi) / sum([x1, x2, ..., xn])
 */
enum activations {
    none,
    relu,
    sigmoid,
    softmax,
};

/**
 * ## Error functions
 * - `mean_squared_error`: f([x1, x2, ..., xn], [e1, e2, ..., en]) = 1 / n * sum_i((xi - ei)^2)
 */
enum error_funcs {
    mean_squared_error
};

class shape2D
{
public:
    size_t x, y;
    shape2D() : x(0), y(0) {}

    /**
     * @brief Set the shape, (y, x) (rows, columns, in matrix)
     */
    shape2D(size_t x, size_t y) : x(x), y(y) {}
    shape2D(const shape2D& other) {*this = other;}

    shape2D& operator=(const shape2D& other) 
    {
        this->x = other.x; this->y = other.y;
        return *this;
    }

    friend bool operator==(const shape2D& lhs, const shape2D& rhs) {
        return lhs.x == rhs.x && lhs.y == rhs.y;
    }

    friend bool operator!=(const shape2D& lhs, const shape2D& rhs) {
        return !(lhs == rhs);
    }

    // Operator for printing to output stream
    friend std::ostream& operator<<(std::ostream& out, const shape2D& shape) {
        return out << shape.x << 'x' << shape.y;
    }
};

END_BLUST_NAMESPACE