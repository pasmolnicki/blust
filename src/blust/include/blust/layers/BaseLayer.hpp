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
    bool    m_built             = false;
    bool   m_in_training        = false;
    size_t m_inputs_size        = 0;
    size_t m_output_size        = 0;
    shape2D m_output_shape      = {};
    BaseLayer* m_next           = nullptr;
    BaseLayer* m_prev           = nullptr;
    matrix_t m_activations      = {};
public:
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
    }

    virtual ~BaseLayer() = default;

    // Get the shape of the output matrix
    shape2D dim() { return m_output_shape; }

    // Set training mode, affects the traning, (for example the wheter to use dropout)
    void set_traning_mode(bool training = true) { m_inputs_size = training; }

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
    virtual matrix_t& feed_forward(matrix_t& inputs) = 0;

    // Feed forward the inputs
    matrix_t& operator()(matrix_t& inputs) { return feed_forward(inputs); }

    // Get the activations of the layer
    matrix_t& get_activations() { return m_activations; }
};


// Base class for learning layers, has `apply` and `gradient` functions
// To modify the weights and biases of the layer during backprop
class BaseLearningLayer : public BaseLayer
{
public:
    BaseLearningLayer() = default;
    BaseLearningLayer(const BaseLearningLayer& other) : BaseLayer(other) {}
    BaseLearningLayer(BaseLearningLayer&& other) : BaseLayer(std::forward<BaseLayer>(other)) {}

    virtual void apply(number_t learning_rate = 0.2, size_t batch_size = 1) = 0;
    virtual void gradient(matrix_t& inputs, matrix_t& expected, error_functor_t& func) = 0;
    virtual void gradient(matrix_t& inputs) = 0;
    virtual number_t cost(matrix_t& expected, error_functor_t& error) = 0;
};


// Base class for weighted layers, has virtual methods for getting the weights, partial derivatives, and weighted inputs
class BaseWeightedLayer : public BaseLearningLayer
{
protected:
    bool m_initialized_weights = false;
public:
    friend class Model;

    BaseWeightedLayer() = default;
    BaseWeightedLayer(const BaseWeightedLayer& other) : BaseLearningLayer(other) {}
    BaseWeightedLayer(BaseWeightedLayer&& other) : BaseLearningLayer(std::forward<BaseLearningLayer>(other)) {}

    virtual void randomize(uint64_t seed = 0x27) = 0;
    virtual matrix_t& get_weights() = 0;
    virtual matrix_t& get_partial_deriv() = 0;
    virtual matrix_t& get_weighted_input() = 0;
    virtual matrix_t& get_gradient_w() = 0;
};

// Fully connected layer, with weights and biases
class WeightedLayer : public BaseWeightedLayer
{
protected:
    matrix_t m_weights;
    matrix_t m_d_weights;
    matrix_t m_weighted_input;
    matrix_t m_partial_deriv;
public:

    friend class Dense;

    WeightedLayer() = default;
    WeightedLayer(const WeightedLayer& other) : BaseWeightedLayer(other)
    {
        m_weights           = other.m_weights;
        m_d_weights         = other.m_d_weights;
        m_weighted_input    = other.m_weighted_input;
        m_partial_deriv     = other.m_partial_deriv;
    }

    WeightedLayer(WeightedLayer&& other) : BaseWeightedLayer(std::forward<BaseWeightedLayer>(other))
    {
        m_weights           = std::move(other.m_weights);
        m_d_weights         = std::move(other.m_d_weights);
        m_weighted_input    = std::move(other.m_weighted_input);
        m_partial_deriv     = std::move(other.m_partial_deriv);
    }

    matrix_t& get_weights() override { return m_weights; }
    matrix_t& get_partial_deriv() override { return m_partial_deriv; }
    matrix_t& get_weighted_input() override { return m_weighted_input; }
    matrix_t& get_gradient_w() override { return m_d_weights; }
};

END_BLUST_NAMESPACE