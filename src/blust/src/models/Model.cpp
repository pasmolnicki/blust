#include <blust/models/Model.hpp>


START_BLUST_NAMESPACE

void Model::backprop(matrix_t& inputs, matrix_t& expected)
{
    auto layer = dynamic_cast<BaseLearningLayer*>(m_output_layer); 
    auto prev  = dynamic_cast<BaseLearningLayer*>(m_output_layer->m_prev);
    auto prev_input = &prev->m_activations;

    // Calculate the output gradient
    layer->gradient(*prev_input, expected, m_error_func);

    while (prev != nullptr)
    {
        layer = prev;
        prev  = dynamic_cast<BaseLearningLayer*>(prev->m_prev);

        if (layer == m_input_layer) // that's the output layer
            prev_input = &inputs;
        else
            prev_input = &prev->m_activations;
        
        layer->gradient(*prev_input);
    }
}

/**
 * @brief Prepare the model for training
 * @brief learning_rate: The learning rate to be used in the model
 */
void Model::compile(number_t learning_rate, error_funcs loss)
{
    m_learning_rate = learning_rate;
    m_error_func.reset(get_error_functor(loss));

    BLUST_ASSERT(dynamic_cast<Input*>(m_input_layer) != nullptr);
    BLUST_ASSERT(dynamic_cast<BaseLearningLayer*>(m_input_layer->m_next) != nullptr);
    BLUST_ASSERT(dynamic_cast<BaseLearningLayer*>(m_output_layer) != nullptr);
}

void Model::fit()
{
    
}

void Model::train_on_batch(batch_t& inputs, batch_t& expected)
{
    // Backpropagate the gradients
    for (size_t i = 0; i < inputs.size(); i++) {
        call(inputs[i]);
        backprop(inputs[i], expected[i]);
    }

    // Apply the gradients
    BaseLearningLayer *layer = 
        dynamic_cast<BaseLearningLayer*>(m_input_layer->m_next);

    while (layer != nullptr)
    {
        layer->apply(m_learning_rate);
        layer = dynamic_cast<BaseLearningLayer*>(layer->m_next);
    }
}

END_BLUST_NAMESPACE