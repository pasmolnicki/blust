#include <blust/models/Model.hpp>


START_BLUST_NAMESPACE

/**
 * @brief Prepare the model for training
 * @param optimizer: The optimizer to be used in the model
 * @param loss: The loss function to be used in the model
 */
void Model::compile(Optimizer* optimizer, error_funcs loss)
{
    m_optimizer.reset(optimizer);
    m_error_func.reset(get_error_functor(loss));
    m_steps = 0;


    // Assert correct layer connections and types
    BLUST_ASSERT(dynamic_cast<Input*>(m_input_layer) != nullptr);
    BLUST_ASSERT(m_input_layer->m_next != nullptr);
    
    BaseLearningLayer* layer = dynamic_cast<BaseLearningLayer*>(m_input_layer->m_next);
    BLUST_ASSERT(layer != nullptr);
    layer->set_optimizer(optimizer->copy());

	// Initialize the weights (if that's a weighted layer)
	BaseWeightedLayer* w_layer = dynamic_cast<BaseWeightedLayer*>(layer);
    if (w_layer != nullptr && !w_layer->m_initialized_weights)
        w_layer->randomize();

    while(true)
    {
        // If that's the last layer, it has to be the output layer
        if (layer->m_next == nullptr)
        {
            BLUST_ASSERT(layer == m_output_layer);
            break;
        }
        
        // Traverse through the layer list
        auto next = dynamic_cast<BaseLearningLayer*>(layer->m_next);
        BLUST_ASSERT(next != nullptr);
        BLUST_ASSERT(next->m_prev == layer);
		next->set_optimizer(optimizer->copy());

        // Initialize the weights (if that's a weighted layer)
        w_layer = dynamic_cast<BaseWeightedLayer*>(layer);
        if (w_layer != nullptr && !w_layer->m_initialized_weights)
            w_layer->randomize();

        layer = next;
    }
}

void Model::call(matrix_t& inputs)
{
    BaseLayer* next     = m_input_layer;
    matrix_t* p_inputs  = &inputs; // avoid too much copying

    while (next != nullptr) {
        p_inputs = &next->feed_forward(*p_inputs);
        next = next->m_next;
    }
}

void Model::backprop(matrix_t& expected)
{
    auto layer      = dynamic_cast<BaseLearningLayer*>(m_output_layer);
    auto prev       = dynamic_cast<BaseLearningLayer*>(m_output_layer->m_prev);
	auto prev_input = &m_output_layer->m_prev->m_activations;

    // Calculate the output gradient
    layer->gradient(*prev_input, expected, m_error_func);

    while (prev != nullptr)
    {
        layer       = prev;
        prev        = dynamic_cast<BaseLearningLayer*>(layer->m_prev);
        prev_input  = &layer->m_prev->m_activations;
        layer->gradient(*prev_input);
    }
}

void Model::fit(batch_t& inputs, batch_t& expected, size_t batch_size)
{
	BLUST_ASSERT(inputs.size() == expected.size());

	size_t n_batches = inputs.size() / batch_size;
	size_t rest      = inputs.size() % batch_size;

	for (size_t i = 0; i < n_batches; i++)
	{
		batch_t batch_input(inputs.begin() + i * batch_size, inputs.begin() + (i + 1) * batch_size);
		batch_t batch_expected(expected.begin() + i * batch_size, expected.begin() + (i + 1) * batch_size);
		train_on_batch(batch_input, batch_expected);

		// Show the cost
		std::cout << "batch " << i + 1 << " loss=" << m_loss_value << '\n';
		m_loss_value = 0;
	}

	if (rest > 0)
	{
		batch_t batch_input(inputs.begin() + n_batches * batch_size, inputs.end());
		batch_t batch_expected(expected.begin() + n_batches * batch_size, expected.end());
		train_on_batch(batch_input, batch_expected);

		// Show the cost
        std::cout << "batch " << n_batches << " loss=" << m_loss_value << '\n';
        m_loss_value = 0;
	}
}

void Model::apply_gradients(size_t steps, size_t batch_size)
{
	BaseLearningLayer* layer =
		dynamic_cast<BaseLearningLayer*>(m_input_layer->m_next);
	while (layer != nullptr)
	{
		layer->apply(m_optimizer->get_decay()->get_learning_rate(steps), batch_size);
		layer = dynamic_cast<BaseLearningLayer*>(layer->m_next);
	}
}

void Model::train_on_batch(batch_t& inputs, batch_t& expected)
{
    // Backpropagate the gradients
    for (size_t i = 0; i < inputs.size(); i++) {
        call(inputs[i]);
        backprop(expected[i]);
		m_loss_value += dynamic_cast<BaseLearningLayer*>(m_output_layer)->cost(expected[i], m_error_func);
        m_steps++;
    }

    // Apply the gradients
	apply_gradients(m_steps, inputs.size());
	m_loss_value /= inputs.size();
    //std::cout << "cost=" << dynamic_cast<BaseLearningLayer*>(m_output_layer)->cost(expected[expected.size() - 1], m_error_func) << '\n';
}

END_BLUST_NAMESPACE