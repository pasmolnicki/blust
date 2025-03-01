#pragma once

#include "BaseLayer.hpp"

START_BLUST_NAMESPACE

// Has fully connected layer of weight to input
class Dense : public WeightedLayer
{
    // Build the layer, with attachment
    void attach_build(BaseLayer& prev_layer)
    {
        attach(&prev_layer);
        build(prev_layer.dim(), m_act_type, mean_squared_error);
    }

    // Activation functions
    activations m_act_type          = none;
    base_function_t m_func_activ    = nullptr;
    base_function_t m_func_deriv    = nullptr;

    // Derivatives
    matrix_t m_biases               = {};
    matrix_t m_d_biases             = {};

public:
	friend class Sequential;
    friend class Model;

    Dense(size_t n_outputs) { m_output_size = n_outputs; }

    // Create the dense layer with activation specified
    Dense(size_t n_outputs, activations act)
    {
        m_output_size = n_outputs;
        m_act_type = act;
    }

    // Build the dense layer, and attach the input layer
    Dense& operator()(BaseLayer& prev) 
    {
        attach_build(prev);
        return *this;
    }

    Dense(const Dense& other) : WeightedLayer(other)
    {
        m_biases        = other.m_biases;
        m_d_biases      = other.m_d_biases;
        m_func_activ    = other.m_func_activ;
        m_func_deriv    = other.m_func_deriv; 
		m_act_type      = other.m_act_type;
    }

    Dense(Dense&& other) : WeightedLayer(std::forward<WeightedLayer>(other))
    {
        m_biases        = std::move(other.m_biases);
        m_d_biases      = std::move(other.m_d_biases);
        m_func_activ    = other.m_func_activ;
        m_func_deriv    = other.m_func_deriv; 
        m_act_type      = other.m_act_type;
    }

    // Build the Dense layer (allocates the memory for matrices)
    void build(shape2D input_shape, activations type = none, error_funcs err = mean_squared_error) 
    {
        m_built         = true;
        m_inputs_size   = input_shape.y;
        m_output_shape  = {input_shape.x, m_output_size};
        auto funcs      = get_functions(type);
        m_func_activ    = funcs.activ;
        m_func_deriv    = funcs.deriv;

        m_weights.build({input_shape.y, m_output_size});
        m_d_weights.build({input_shape.y, m_output_size});
        m_biases.build({input_shape.x, m_output_size});
        m_d_biases.build({input_shape.x, m_output_size});
    }

    // Set random values to weights and baises, with given seed
    void randomize(uint64_t seed = 0x27) override 
    {
        m_initialized_weights = true;
        utils::randomize(m_weights.begin(), m_weights.end(), m_inputs_size, seed);
        utils::randomize(m_biases.begin(), m_biases.end(), m_inputs_size, seed);
    }

    // Calculate the hidden gradient, expects next layer to be a child of `BaseDense` (or none)
    void gradient(matrix_t& inputs) override
    {
        // Calculate the partial derivative:
        // N_(L-1) = P_(L) * W_(L).T
        // P_(L-1) = N_(L-1) % A_(L-1)
        // then dC/dW_(L-1) = A_(L-2).T * P_(L-1)
        auto next       = dynamic_cast<WeightedLayer*>(m_next);
        auto N          = next->m_partial_deriv * next->m_weights.T();
        auto dA         = m_func_deriv(m_activations);
        m_partial_deriv = N % dA;

        m_d_weights     += inputs.T() * m_partial_deriv;
        m_d_biases      += m_partial_deriv;
    }

    // Calculate the output gradient
    void gradient(matrix_t& inputs, matrix_t& expected, error_functor_t& func) override
    {
        auto dA          = m_func_deriv(m_activations);
        auto dC          = func->d_cost(m_activations, expected);
        m_partial_deriv  = dA % dC;
        m_d_weights     += inputs.T() * m_partial_deriv;
        m_d_biases      += m_partial_deriv;
    }

    void apply(number_t learning_rate = 0.2, size_t batch_size = 1)
    {
        m_weights -= m_d_weights * (learning_rate / batch_size);
        m_biases -= m_d_biases * (learning_rate / batch_size);

		m_d_weights.fill(0);
		m_d_biases.fill(0);
    }

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

    // Get the total cost of the network, based on the specified error function
    number_t cost(matrix_t& expected, error_functor_t& error) override
    {
        return error->error(m_activations, expected);
    }

    matrix_t& get_biases() { return m_biases; }
    matrix_t& get_gradient_b() { return m_d_biases; }
};

END_BLUST_NAMESPACE