#pragma once

#include <blust/macros.hpp>
#include <blust/base_types.hpp>
#include <blust/tensor.hpp>
#include <blust/types.hpp>
#include <blust/functions.hpp>

START_BLUST_NAMESPACE

class Dense;

// Specific neural network operations
class nn_ops
{
public:

    virtual void nn_hidden_gradient(Dense* layer);
    virtual void nn_output_gradient(Dense* layer, tensor_t& expected, error_functor_t& func);
    virtual void nn_feed_forward(Dense* layer, tensor_t& inputs);
};

END_BLUST_NAMESPACE