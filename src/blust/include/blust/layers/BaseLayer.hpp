#pragma once

#include <blust/types.hpp>
#include <blust/functions.hpp>

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
    bool   m_in_training = false;
    size_t m_inputs_size = 0;
    size_t m_output_size = 0;
public:
    BaseLayer() = default;

    // Allocate memory for the weights, and set activation function
    virtual void build(shape2D input_shape, activations func) = 0;

    // Set training mode, affects the traning, (for example the wheter to use dropout)
    void set_traning_mode(bool training = true) { m_inputs_size = training; }

    // Set random values for the weights
    virtual void randomize(uint64_t seed = 0x27) = 0;

    // Get derivative (depends on the context)
    virtual matrix_t& deriv() = 0;

    // Calculate layer activations
    virtual matrix_t& feed_forward(matrix_t& inputs) = 0;

    // Get saved outputs after `feed_forward`
    virtual matrix_t& get_activations() = 0;
};

class BaseDense : public BaseLayer
{
public:
    BaseDense(size_t n_outputs)
    {
        m_output_size = n_outputs;
    }

    // Build the Dense layer (allocates the memory for matrices)
    void build(shape2D input_shape, activations type = relu) override 
    {
        m_inputs_size = input_shape.y;
        auto funcs = get_functions(type);
        m_func_activ = funcs.activ;
        m_func_deriv = funcs.deriv;

        m_weights.build({input_shape.y, m_output_size});
        m_d_weights.build({input_shape.y, m_output_size});
        m_biases.build({input_shape.x, m_output_size});
        m_d_biases.build({input_shape.x, m_output_size});
    }

    // Set random values to weights and baises, with given seed
    void randomize(uint64_t seed = 0x27) override 
    {
        utils::randomize(m_weights.begin(), m_weights.end(), m_inputs_size, seed);
        utils::randomize(m_biases.begin(), m_biases.end(), m_inputs_size, seed);
    }

    matrix_t& deriv() override { return m_partial_deriv; }

    /**
     * @brief Multiply the inputs by weights and add biases, return the result
     * @throw `InvalidMatrixSize` if the specified inputs are the same dimensions as specified in `build`
     */
    matrix_t& feed_forward(matrix_t& inputs) override
    {
        m_weighted_input = inputs * m_weights;
        m_weighted_input += m_biases;
        m_activations    = m_func_activ(m_weighted_input);
        return m_activations;
    }

    // Get saved outputs after `feed_forward`
    matrix_t& get_activations() override { return m_activations; }
    matrix_t& get_weighted_input() { return m_weighted_input; }
    matrix_t& get_weights() { return m_weights; }
    matrix_t& get_biases() { return m_biases; }
    matrix_t& get_gradient_w() { return m_d_weights; }
    matrix_t& get_gradient_b() { return m_d_biases; }

protected:
    base_function_t m_func_activ;
    base_function_t m_func_deriv;

    matrix_t m_activations;
    matrix_t m_weighted_input;
    matrix_t m_partial_deriv;

    // Derivatives
    matrix_t m_d_weights;
    matrix_t m_d_biases;

    // Weights, biases
    matrix_t m_weights;
    matrix_t m_biases;
};

END_BLUST_NAMESPACE