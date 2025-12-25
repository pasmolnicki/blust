#pragma once

#include "optimizer.hpp"

#include <memory>


START_BLUST_NAMESPACE

// Stochastic Gradient Descent Optimizer
class SGD : public Optimizer
{
private:

	// typedef void(*update_func_t)(tensor_t&, tensor_t&, tensor_t&, number_t, number_t);
	typedef void(SGD::*update_func_t)(
		tensor_t& weighted_grad, tensor_t& grad, tensor_t& w, number_t learning_rate, tensor_t& velocity
	);

	// Number of steps taken
	size_t m_step{0};
	// Momentum factor [0.0, 1.0)
	number_t m_momentum{0.0};
	// Enable Nesterov momentum
	bool m_nesterov{false};
	// Gradient clipping
	number_t m_clipnorm;
	// Gradient clipping by value
	number_t m_clipvalue;
	// Velocity weight tensor
	tensor_t m_velocity_w;
	// Velocity bias tensor
	tensor_t m_velocity_b;
	// The update function, either with momentum or without
	update_func_t m_updater;

	// Temporary weighted gradient tensor
	tensor_t m_weighted_grad_w;
	tensor_t m_weighted_grad_b;

	// If nestrov is enabled, update the weights with nestrov
	void M_update_nestrov(
		tensor_t& weighted_grad, tensor_t& grad, tensor_t& w, number_t learning_rate, tensor_t& velocity
	)
	{
		// velocity = momentum * velocity - learning_rate * grad;
		// w		+= momentum * velocity - learning_rate * grad;
	}

	// Update the weights without nestrov (momentum is larger than 0)
	void M_update_momentum(
		tensor_t& weighted_grad, tensor_t& grad, tensor_t& w, number_t learning_rate, tensor_t& velocity
	)
	{
		ops->mul(velocity, m_momentum, velocity);
		ops->mul(grad, learning_rate, weighted_grad);
		ops->sub(velocity, weighted_grad, velocity);
		ops->add(w, velocity, w);
	}

	// Update the weights without momentum
	void M_update(
		tensor_t& weighted_grad, tensor_t& grad, tensor_t& w, number_t learning_rate, tensor_t& /*velocity*/
	)
	{
		ops->mul(grad, learning_rate, weighted_grad);
		ops->sub(w, weighted_grad, w);
	}

	// The updater function
	void M_set_updater()
	{
		if (m_momentum > 0)
		{
			if (m_nesterov)
				m_updater = &SGD::M_update_nestrov;
			else
				m_updater = &SGD::M_update_momentum;
		}
		else
			m_updater = &SGD::M_update;
	}

public:

	SGD(
		const SGD& other
	) : Optimizer(other), m_momentum(other.m_momentum), m_nesterov(other.m_nesterov),
		m_clipnorm(other.m_clipnorm), m_clipvalue(other.m_clipvalue) 
	{
		M_set_updater();
	}

	SGD& operator=(const SGD& other)
	{
		(void)Optimizer::operator=(other);
		m_momentum		= other.m_momentum;
		m_nesterov		= other.m_nesterov;
		m_clipnorm		= other.m_clipnorm;
		m_clipvalue		= other.m_clipvalue;
		m_velocity_b	= other.m_velocity_b;
		m_velocity_w	= other.m_velocity_w;
		M_set_updater();
		return *this;
	}

	SGD(
		number_t learning_rate = 1e-2,
		number_t momentum = 0.0,
		bool nesterov = false,
		number_t clipnorm = 0.0,
		number_t clipvalue = 0.0
	) : m_momentum(momentum), m_nesterov(nesterov),
		m_clipnorm(clipnorm), m_clipvalue(clipvalue) 
	{
		m_decay = std::make_shared<ConstantDecay>(learning_rate);
		M_set_updater();
	}

	SGD(
		BaseDecay* decay,
		number_t momentum = 0.0,
		bool nesterov = false,
		number_t clipnorm = 0.0,
		number_t clipvalue = 0.0
	) : m_momentum(momentum), m_nesterov(nesterov),
		m_clipnorm(clipnorm), m_clipvalue(clipvalue) 
	{
		m_decay = std::shared_ptr<BaseDecay>(decay);
		M_set_updater();
	}

	void build(shape wdim, shape bdim) override
	{
		m_velocity_w = tensor_t(wdim);
		m_velocity_b = tensor_t(bdim);
		m_weighted_grad_w = tensor_t(wdim);
		m_weighted_grad_b = tensor_t(bdim);
	}

	// Update the weights and biases
	void update_step(tensor_t& grad_w, tensor_t& grad_b, tensor_t& w, tensor_t& b, number_t learning_rate) override
	{
		std::invoke(m_updater, this, m_weighted_grad_w, grad_w, w, learning_rate, m_velocity_w);
		std::invoke(m_updater, this, m_weighted_grad_b, grad_b, b, learning_rate, m_velocity_b);
	}

	// Update the weights
	void update_step(tensor_t& grad_w, tensor_t& w, number_t learning_rate) override
	{
		std::invoke(m_updater, this, m_weighted_grad_w, grad_w, w, learning_rate, m_velocity_w);
	}

	// Copy the optimizer
	Optimizer* copy() override
	{
		return new SGD(*this);
	}
};


END_BLUST_NAMESPACE