#pragma once

#include "Model.hpp"

START_BLUST_NAMESPACE


class Sequential : public Model
{
	std::vector<std::shared_ptr<BaseLayer>> m_layers;

	inline void M_add_dense(Dense* layer)
	{
		if (!m_layers.empty())
			layer->attach_build(*m_layers.back());
		m_layers.emplace_back(layer);
	}

public:
	Sequential() = default;

	// Construct a model with a list of layers (all layers should be pointers, allocated on the heap)
	Sequential(std::initializer_list<BaseLayer*> layers)
	{
		m_layers.reserve(layers.size());
		for (auto layer : layers)
		{
			auto dense = dynamic_cast<Dense*>(layer);
			if (dense != nullptr)
			{
				M_add_dense(dense);
			}
			else
				m_layers.emplace_back(layer);
		}
	}
	
	// Push new layer to the stack
	void add(Dense&& layer) {
		if (!m_layers.empty())
			layer.attach_build(*m_layers.back());
		m_layers.emplace_back(new Dense(std::forward<Dense>(layer)));

		// TODO: fix this bug, the traning is not working, when adding layer like this:
		//M_add_dense(new Dense(std::forward<Dense>(layer)));
	}

	// Add input layer to the stack
	void add(Input&& layer) {
		m_layers.emplace_back(new Input(std::forward<Input>(layer)));
	}

	// Prepare the model for learning
	void compile(Optimizer* optimizer, error_funcs loss = mean_squared_error) override
	{
		m_input_layer = m_layers.front().get();
		m_output_layer = m_layers.back().get();

		Model::compile(optimizer, loss);
	}
};


END_BLUST_NAMESPACE