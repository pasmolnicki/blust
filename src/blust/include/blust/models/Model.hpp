#pragma once

#include <blust/layers/BaseLayer.hpp>
#include <blust/layers/Input.hpp>

START_BLUST_NAMESPACE

class Model
{
public:
    Model(BaseLayer* inputs, BaseLayer* outputs) : 
        m_input_layer(inputs), m_output_layer(outputs) {}

    void compile(number_t learning_rate = 0.2, error_funcs loss = mean_squared_error);

    // Feed forward the model
    void call(matrix_t& inputs)
    {
        BaseLayer *next     = m_input_layer;
        matrix_t *p_inputs  = &inputs; // avoid too much copying

        while (next != nullptr) {
            p_inputs = &next->feed_forward(*p_inputs);
            next     = next->m_next;
        }
    }

    // Get predictions from the model
    matrix_t& predict(matrix_t& inputs)
    {
        call(inputs);
        return m_output_layer->m_activations;
    }
    
    // Backpropagte on a single data input
    void backprop(matrix_t& inputs, matrix_t& expected);
    
    virtual void fit();
    void train_on_batch(batch_t& inputs, batch_t& expected);

protected:
    BaseLayer* m_input_layer;
    BaseLayer* m_output_layer;

    number_t m_learning_rate     = 0.2;
    error_functor_t m_error_func = nullptr;
};


END_BLUST_NAMESPACE