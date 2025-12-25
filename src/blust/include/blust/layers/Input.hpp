#pragma once

#include "BaseLayer.hpp"

START_BLUST_NAMESPACE

// Has fully connected layer of weight to input
class Input : public BaseLayer
{
public:
    Input(shape shape) { 
        m_output_shape  = shape; 
        m_output_size   = 0;
        m_inputs_size   = 0;
        m_activations.build(shape);
        m_transp_activations.build(shape.T());
    }

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
    tensor_t& feed_forward(tensor_t& inputs) override
    {
        m_activations = inputs;
        ops->transpose(m_activations, m_transp_activations);
        return m_activations;
    }

    virtual size_t bytesize() const override {
        return BaseLayer::bytesize() + sizeof(*this);
    }
};

END_BLUST_NAMESPACE