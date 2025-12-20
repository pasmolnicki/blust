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
    tensor_t m_biases               = {};
    tensor_t m_d_biases             = {};

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
    void build(shape input_shape, activations type = none, error_funcs err = mean_squared_error) 
    {
        m_built         = true;
        m_inputs_size   = input_shape.dim()[1];
        m_output_shape  = {input_shape.dim()[0], m_output_size};
        auto funcs      = get_functions(type);
        m_func_activ    = funcs.activ;
        m_func_deriv    = funcs.deriv;

		if (m_initialized_weights)
			return;

        if (m_optimizer)
			m_optimizer->build({ m_inputs_size, m_output_size }, m_output_shape);

        m_weights.build({ m_inputs_size, m_output_size});
        m_d_weights.build({ m_inputs_size, m_output_size});
        m_biases.build(m_output_shape);
        m_d_biases.build(m_output_shape);
    }

    // Set random values to weights and baises, with given seed
    void randomize(uint64_t seed = 0x27) override 
    {
        m_initialized_weights = true;
        utils::randomize(m_weights, seed);
        utils::randomize(m_biases, seed);
        // utils::randomize(m_weights.begin(), m_weights.end(), m_inputs_size, seed);
        // utils::randomize(m_biases.begin(), m_biases.end(), m_inputs_size, seed);
    }

    // Calculate the hidden gradient, expects next layer to be a child of `BaseDense` (or none)
    void gradient(tensor_t& inputs) override
    {
        // Calculate the partial derivative:
        // N_(L-1) = P_(L) * W_(L).T
        // P_(L-1) = N_(L-1) % A_(L-1)
        // then dC/dW_(L-1) = A_(L-2).T * P_(L-1)
        auto next       = dynamic_cast<WeightedLayer*>(m_next);
        auto N          = ops->mat_mul(next->m_partial_deriv, ops->transpose(next->m_weights));
        auto dA         = m_func_deriv(m_activations);
        m_partial_deriv = ops->hadamard(N, dA);

        auto ops_weights = ops_tensor(m_d_weights); // Shares the buffer with 'm_d_weights'
        auto ops_biases  = ops_tensor(m_d_biases);  // Shares the buffer
        auto w_grad = ops->mat_mul(ops->transpose(inputs), m_partial_deriv);
        ops->add(ops_weights, w_grad, ops_weights);
        auto b_grad = ops_tensor(m_partial_deriv);
        ops->add(ops_biases, b_grad, ops_biases);
    }

    // Calculate the output gradient
    void gradient(tensor_t& inputs, tensor_t& expected, error_functor_t& func) override
    {
        auto dA          = m_func_deriv(m_activations);
        auto dC          = func->d_cost(m_activations, expected);
        m_partial_deriv  = ops->hadamard(dA, dC);
        auto w_grad      = ops->mat_mul(ops->transpose(inputs), m_partial_deriv);

        auto ops_weights = ops_tensor(m_d_weights); // Shares the buffer with 'm_d_weights'
        auto ops_biases  = ops_tensor(m_d_biases);  // Shares the buffer with 'm_d_biases'
        auto ops_partial = ops_tensor(m_partial_deriv); // Shares the buffer with 'm_partial_deriv'
        ops->add(ops_weights, w_grad, ops_weights);
        ops->add(ops_biases, ops_partial, ops_biases);
    }

    void apply(number_t learning_rate = 0.2, size_t batch_size = 1)
    {
        auto ops_weights = ops_tensor(m_d_weights); // Shares the buffer with 'm_d_weights'
        auto ops_biases  = ops_tensor(m_d_biases);  // Shares the buffer

		ops->div(ops_weights, static_cast<number_t>(batch_size), ops_weights);
        ops->div(ops_biases, static_cast<number_t>(batch_size), ops_biases);

		m_optimizer->update_step(m_d_weights, m_d_biases, m_weights, m_biases, learning_rate);

        /*m_weights -= m_d_weights * (learning_rate / batch_size);
        m_biases -= m_d_biases * (learning_rate / batch_size);*/

		m_d_weights.fill(0);
		m_d_biases.fill(0);
    }

    /**
     * @brief Multiply the inputs by weights and add biases, return the result
     * @throw `InvalidMatrixSize` if the specified inputs are the same dimensions as specified in `build`
     */
    tensor_t& feed_forward(tensor_t& inputs) override
    {
        // m_weighted_input = inputs * m_weights;
        // m_weighted_input += m_biases;
        m_weighted_input = ops->add(m_biases, ops->mat_mul(inputs, m_weights));
        m_activations    = m_func_activ(m_weighted_input);
        return m_activations;
    }

    // Get the total cost of the network, based on the specified error function
    number_t cost(tensor_t& expected, error_functor_t& error) override
    {
        return error->error(m_activations, expected);
    }

    tensor_t& get_biases() { return m_biases; }
    tensor_t& get_gradient_b() { return m_d_biases; }
};

END_BLUST_NAMESPACE