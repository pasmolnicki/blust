#pragma once

#include "BaseLayer.hpp"

START_BLUST_NAMESPACE

// Has fully connected layer of weight to input
class Input : public BaseLayer
{
public:
    Input(shape2D shape) { m_output_shape = shape; }

    Input(const Input& other) : BaseLayer(other)
    {
        m_output_shape = other.m_output_shape;
        m_activations  = other.m_activations;
    }

    Input(Input&& other) : BaseLayer(std::forward<BaseLayer>(other))
    {
        m_output_shape = other.m_output_shape;
        m_activations  = std::move(other.m_activations);
    }

    // Set the `activations`
    matrix_t& feed_forward(matrix_t& inputs) override
    {
        m_activations = inputs;
        return m_activations;
    }
};

END_BLUST_NAMESPACE