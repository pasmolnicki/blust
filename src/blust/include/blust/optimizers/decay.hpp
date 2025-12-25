#pragma once

#include <blust/base_types.hpp>

START_BLUST_NAMESPACE

class BaseDecay
{
public:
	BaseDecay() = default;
	virtual ~BaseDecay() = default;
	virtual number_t get_learning_rate(size_t step) = 0;
};


class ConstantDecay : public BaseDecay
{
private:
	number_t m_learning_rate;
public:
	ConstantDecay(number_t rate = 0.1) : m_learning_rate(rate) {}
	number_t get_learning_rate(size_t step) override
	{
		return m_learning_rate;
	}
};


class ExponentialDecay : public BaseDecay
{
private:
	number_t m_learning_rate;
	number_t m_decay;
	size_t m_decay_steps;
public:

	ExponentialDecay(
		number_t rate = 0.1,
		number_t decay = 0.96,
		size_t decay_steps = 1000
	) : m_learning_rate(rate), m_decay(decay), m_decay_steps(decay_steps) {}

	
	number_t get_learning_rate(size_t step) override
	{
		return m_learning_rate * std::pow(m_decay, static_cast<number_t>(step) / m_decay_steps);
	}
};






END_BLUST_NAMESPACE