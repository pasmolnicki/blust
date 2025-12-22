#pragma once

#include <blust/types.hpp>
#include <blust/backend/operations.hpp>
#include "decay.hpp"

START_BLUST_NAMESPACE

// Base class for all optimizers
class Optimizer
{
protected:
	std::shared_ptr<BaseDecay> m_decay;
public:
	Optimizer() = default;
	Optimizer(const Optimizer& other)
	{
		m_decay = other.m_decay;
	}

	Optimizer& operator=(const Optimizer& other) noexcept
	{
		m_decay = other.m_decay;
		return *this;
	}

	virtual ~Optimizer() = default;

	// Create the Optimizer
	virtual void build(shape w_dim, shape b_dim) = 0;
	
	// Update the weights and biases with given gradients and learning rate
	virtual void update_step(tensor_t& grad_w, tensor_t& grad_b, tensor_t& w, tensor_t& b, number_t learning_rate) = 0;

	// Update the weights with given gradients and learning rate
	virtual void update_step(tensor_t& grad_w, tensor_t& w, number_t learning_rate) = 0;

	// Create a copy of the optimizer
	virtual Optimizer* copy() = 0;

	// Get the decay strategy
	std::shared_ptr<BaseDecay>& get_decay() { return m_decay; }

	// Get the learning rate at given step
	number_t get_learning_rate(size_t step) {
		return m_decay->get_learning_rate(step);
	}
};

END_BLUST_NAMESPACE