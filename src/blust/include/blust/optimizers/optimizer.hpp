#pragma once

#include <blust/types.hpp>
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
	virtual void build(shape2D w_dim, shape2D b_dim) = 0;
	virtual void update_step(matrix_t& grad_w, matrix_t& grad_b, matrix_t& w, matrix_t& b, number_t learning_rate) = 0;
	virtual void update_step(matrix_t& grad_w, matrix_t& w, number_t learning_rate) = 0;
	virtual Optimizer* copy() = 0;
	std::shared_ptr<BaseDecay>& get_decay() { return m_decay; }
};

END_BLUST_NAMESPACE