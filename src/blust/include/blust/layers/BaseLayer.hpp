#pragma once

#include <blust/types.hpp>
#include <blust/functions.hpp>
#include <blust/optimizers/optimizer.hpp>
#include <blust/backend/operations.hpp>

START_BLUST_NAMESPACE

// Types of layers:
// - Dense (input x ouput, should have activation function specified, )
// - Input (input) (not really a layer tho)
// - Output (output, with activation functions)
// - Convolution Layer
// Because the layers will be used as a whole in a model
// I should be able to get the derivatives needed for backpropagation
class BaseLayer
{
protected:

    bool    m_built             = false;
    bool   m_in_training        = false;
    size_t m_inputs_size        = 0;
    size_t m_output_size        = 0;
    shape m_output_shape        = {};
    BaseLayer* m_next           = nullptr;
    BaseLayer* m_prev           = nullptr;
    tensor_t m_activations      = {};
    tensor_t m_transp_activations  = {};

public:
    friend class nn_ops;
    friend class Model;
    friend class Sequential;

    BaseLayer() = default;
    BaseLayer(const BaseLayer& other)
    {
        m_built             = other.m_built;
        m_in_training       = other.m_in_training;
        m_inputs_size       = other.m_inputs_size;
        m_output_size       = other.m_output_size;
        m_output_shape      = other.m_output_shape;
        attach(other.m_prev);
        m_next              = other.m_next;
        m_activations       = other.m_activations;
        m_transp_activations = other.m_transp_activations;
    }

    BaseLayer(BaseLayer&& other)
    {
        m_built             = other.m_built;
        m_in_training       = other.m_in_training;
        m_inputs_size       = other.m_inputs_size;
        m_output_size       = other.m_output_size;
        m_output_shape      = other.m_output_shape;
        attach(other.m_prev);
        m_next              = other.m_next;
        m_activations       = std::move(other.m_activations);
        m_transp_activations = std::move(other.m_transp_activations);
    }

    virtual ~BaseLayer() = default;

    // Get the shape of the output matrix
    shape dim() { return m_output_shape; }

    // Set training mode, affects the traning, (for example the wheter to use dropout)
    void set_traning_mode(bool training = true) {}

    // Attach the input layer
    void attach(BaseLayer* prev) 
    { 
		if (prev == nullptr) return;

        prev->m_next = this; 
        this->m_prev = prev;
    }

    // See if there is no next layer attached, means this is the output layer
    bool last() { return m_next == nullptr; }

    // See if there is no previous layer attached, means this is the input layer
    bool first() { return m_prev == nullptr; }

    // Calculate layer activations
    virtual tensor_t& feed_forward(tensor_t& inputs) = 0;

    // Feed forward the inputs
    tensor_t& operator()(tensor_t& inputs) { return feed_forward(inputs); }

    // Get the activations of the layer
    tensor_t& get_activations() { return m_activations; }

    // Get the transposed activations of the layer
    tensor_t& get_transp_activations() { return m_transp_activations; }

    // Get total memory used by the layer in bytes
    virtual size_t bytesize() const {
        return m_activations.bytesize() + m_transp_activations.bytesize();
    }
};


END_BLUST_NAMESPACE