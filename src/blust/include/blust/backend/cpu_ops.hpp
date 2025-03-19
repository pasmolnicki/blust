#pragma once

#include <xmmintrin.h>

#include "operations.hpp"

START_BLUST_NAMESPACE

class cpu_ops : public operations
{
public:
    cpu_ops() = default;

    // Add 2 tensors
    ops_tensor add(ops_tensor, ops_tensor);
    ops_tensor sub(ops_tensor, ops_tensor);
};

END_BLUST_NAMESPACE