#pragma once

#include "optimizer.hpp"

#include <memory>


START_BLUST_NAMESPACE

// Stochastic Gradient Descent Optimizer
class SGD : public Optimizer
{
private:

	typedef void(*update_func_t)(matrix_t&, matrix_t&, matrix_t&, number_t, number_t);

	number_t m_momentum;
	bool m_nesterov;
	number_t m_clipnorm;
	number_t m_clipvalue;
	matrix_t m_velocity_w;
	matrix_t m_velocity_b;
	update_func_t m_updater;


	// If nestrov is enabled, update the weights with nestrov
	static void M_update_nestrov(
		matrix_t& velocity, matrix_t& grad, matrix_t& w, number_t learning_rate, number_t momentum
	)
	{
		velocity = momentum * velocity - learning_rate * grad;
		w		+= momentum * velocity - learning_rate * grad;
	}

	// Update the weights without nestrov (momentum is larger than 0)
	static void M_update_momentum(
		matrix_t& velocity, matrix_t& grad, matrix_t& w, number_t learning_rate, number_t momentum
	)
	{
		velocity = momentum * velocity - learning_rate * grad;
		w		+= velocity;
	}

	// Update the weights without momentum
	static void M_update(
		matrix_t& _, matrix_t& grad, matrix_t& w, number_t learning_rate, number_t __
	)
	{
		w -= learning_rate * grad;
	}

	// The updater function
	void M_set_updater()
	{
		if (m_momentum > 0)
		{
			if (m_nesterov)
				m_updater = M_update_nestrov;
			else
				m_updater = M_update_momentum;
		}
		else
			m_updater = M_update;
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

	void build(shape2D wdim, shape2D bdim) override
	{
		m_velocity_w = matrix_t::zeros(wdim);
		m_velocity_b = matrix_t::zeros(bdim);
	}

	// Update the weights and biases
	void update_step(matrix_t& grad_w, matrix_t& grad_b, matrix_t& w, matrix_t& b, number_t learning_rate) override
	{
		m_updater(m_velocity_w, grad_w, w, learning_rate, m_momentum);
		m_updater(m_velocity_b, grad_b, b, learning_rate, m_momentum);
	}

	// Update the weights
	void update_step(matrix_t& grad_w, matrix_t& w, number_t learning_rate) override
	{
		m_updater(m_velocity_w, grad_w, w, learning_rate, m_momentum);
	}

	// Copy the optimizer
	Optimizer* copy() override
	{
		return new SGD(*this);
	}
};


END_BLUST_NAMESPACE