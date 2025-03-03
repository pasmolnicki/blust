#pragma once

#include <blust/layers/BaseLayer.hpp>
#include <blust/layers/Input.hpp>
#include <blust/optimizers/all_opt.hpp>

START_BLUST_NAMESPACE

// Base class for all models
class Model
{
public:
    Model() = default;

    Model(BaseLayer* inputs, BaseLayer* outputs) : 
        m_input_layer(inputs), m_output_layer(outputs) {}

	inline void init(BaseLayer* inputs, BaseLayer* outputs)
	{
		m_input_layer = inputs;
		m_output_layer = outputs;
	}

    // Prepare the model for learning
	virtual void compile(Optimizer* optimizer, error_funcs loss = mean_squared_error);

    // Feed forward the model
    void call(matrix_t& inputs);

    // Get predictions from the model
    inline matrix_t& predict(matrix_t& inputs)
    {
        call(inputs);
        return m_output_layer->m_activations;
    }
    
    // Backpropagte on a single data input
    void backprop(matrix_t& expected);
	void apply_gradients(size_t steps, size_t batch_size);
    void fit(batch_t& inputs, batch_t& expected, size_t batch_size = 30);
    void train_on_batch(batch_t& inputs, batch_t& expected);

protected:
    BaseLayer* m_input_layer                = nullptr;
    BaseLayer* m_output_layer               = nullptr;
	std::unique_ptr<Optimizer> m_optimizer  = nullptr;
	number_t m_loss_value                   = number_t(0);
    size_t m_steps                          = 0;
    error_functor_t m_error_func            = nullptr;
};


END_BLUST_NAMESPACE