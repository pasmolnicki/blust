#pragma once

#include <blust/layers/BaseLayer.hpp>

START_BLUST_NAMESPACE

// Base class for learning layers, has `apply` and `gradient` functions
// To modify the weights and biases of the layer during backprop
class BaseLearningLayer : public BaseLayer
{
protected:
    std::unique_ptr<Optimizer> m_optimizer = nullptr;
public:
    
    BaseLearningLayer() = default;
    BaseLearningLayer(const BaseLearningLayer& other) : BaseLayer(other) {}
    BaseLearningLayer(BaseLearningLayer&& other) : BaseLayer(std::forward<BaseLayer>(other)) {}

    virtual void apply(number_t learning_rate = 0.2, size_t batch_size = 1) = 0;
    virtual void gradient(tensor_t& expected, error_functor_t& func) = 0;
    virtual void gradient() = 0;
    virtual number_t cost(tensor_t& expected, error_functor_t& error) = 0;

	// Set the optimizer for the layer
    virtual void set_optimizer(Optimizer* optimizer)
    {
		m_optimizer.reset(optimizer);
		if (m_built)
		    m_optimizer->build({ m_inputs_size, m_output_size }, m_output_shape);
    }

    virtual size_t bytesize() const override {
        return BaseLayer::bytesize();
    }
};


END_BLUST_NAMESPACE