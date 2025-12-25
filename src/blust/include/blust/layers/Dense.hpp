#pragma once

#include <blust/layers/WeightedLayer.hpp>

START_BLUST_NAMESPACE

// Has fully connected layer of weight to input
class Dense : public WeightedLayer
{    
private:
    // Build the layer, with attachment
    void attach_build(BaseLayer& prev_layer)
    {
        attach(&prev_layer);
        build(prev_layer.dim(), m_act_type, mean_squared_error);
    }

    // Activation functions
    activation_type m_act_type          = none;
    base_activation_t m_func_activ     = nullptr;

    // Derivatives
    tensor_t m_biases               = {};
    tensor_t m_d_biases             = {};

public:
    friend class nn_ops;

	friend class Sequential;
    friend class Model;

    Dense(size_t n_outputs) { m_output_size = n_outputs; }

    // Create the dense layer with activation specified
    Dense(size_t n_outputs, activation_type act)
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
		m_act_type      = other.m_act_type;
        m_func_activ    = other.m_func_activ;
    }

    Dense(Dense&& other) : WeightedLayer(std::forward<WeightedLayer>(other))
    {
        m_biases        = std::move(other.m_biases);
        m_d_biases      = std::move(other.m_d_biases);
        m_act_type      = other.m_act_type;
        m_func_activ    = other.m_func_activ;
    }

    // Build the Dense layer (allocates the memory for matrices)
    void build(shape input_shape, activation_type type = none, error_funcs err = mean_squared_error) 
    {
        m_built         = true;
        m_inputs_size   = input_shape.dim()[1];
        m_output_shape  = {input_shape.dim()[0], m_output_size};
        m_func_activ    = get_base_activation(type);

		if (m_initialized_weights)
			return;

        if (m_optimizer)
			m_optimizer->build({ m_inputs_size, m_output_size }, m_output_shape);


        m_transp_weights.build({ m_output_size, m_inputs_size});
        m_weights.build({ m_inputs_size, m_output_size});
        m_d_weights.build({ m_inputs_size, m_output_size});
        m_biases.build(m_output_shape);
        m_d_biases.build(m_output_shape);
        m_activations.build(m_output_shape);
        m_partial_deriv.build(m_output_shape);
        m_weighted_input.build(m_output_shape);
        m_transp_activations.build(m_output_shape.T());
    }

    // Set random values to weights and baises, with given seed
    void randomize(uint64_t seed = 0x27) override 
    {
        m_initialized_weights = true;
        utils::randomize(m_weights, seed);
        utils::randomize(m_biases, seed);

        // Update transposed weights
        ops->transpose(m_weights, m_transp_weights);
    }

    // Calculate the hidden gradient, expects next layer to be a child of `BaseDense` (or none)
    void gradient() override
    {
        // // Calculate the partial derivative:
        // // N_(L-1) = P_(L) * W_(L).T
        // // P_(L-1) = N_(L-1) % A_(L-1)
        // // then dC/dW_(L-1) = A_(L-2).T * P_(L-1)
        // auto next       = dynamic_cast<WeightedLayer*>(m_next);
        // auto N          = ops->mat_mul(next->m_partial_deriv, ops->transpose(next->m_weights));
        // auto dA         = m_func_deriv(m_activations);
        // m_partial_deriv = ops->hadamard(N, dA);
        ops->nn_hidden_gradient(this);
    }

    // Calculate the output gradient
    void gradient(tensor_t& expected, error_functor_t& func) override
    {
        // auto dA          = m_func_deriv(m_activations);
        // auto dC          = func->d_cost(m_activations, expected);
        // m_partial_deriv  = ops->hadamard(dA, dC);
        // auto w_grad      = ops->mat_mul(ops->transpose(inputs), m_partial_deriv);
        ops->nn_output_gradient(this, expected, func);
    }

    void apply(number_t learning_rate = 0.2, size_t batch_size = 1)
    {
		m_optimizer->update_step(m_d_weights, m_d_biases, m_weights, m_biases, learning_rate);

        // Update transposed weights
        ops->transpose(m_weights, m_transp_weights);

		m_d_weights.fill(0);
		m_d_biases.fill(0);
    }

    /**
     * @brief Multiply the inputs by weights and add biases, return the result
     * @throw `InvalidMatrixSize` if the specified inputs are the same dimensions as specified in `build`
     */
    tensor_t& feed_forward(tensor_t& inputs) override
    {
        ops->nn_feed_forward(this, inputs);
        ops->transpose(m_activations, m_transp_activations);
        return m_activations;
    }

    // Get the total cost of the network, based on the specified error function
    number_t cost(tensor_t& expected, error_functor_t& error) override
    {
        return error->error(m_activations, expected);
    }

    tensor_t& get_biases() { return m_biases; }
    tensor_t& get_gradient_b() { return m_d_biases; }

    virtual size_t bytesize() const override {
        return WeightedLayer::bytesize() + m_biases.bytesize() + m_d_biases.bytesize();
    }

    // Get the activation function
    base_activation_t get_activation_fn() { return m_func_activ; }
};

END_BLUST_NAMESPACE