#pragma once

#include "BaseLayer.hpp"

START_BLUST_NAMESPACE

class Output : public BaseDense
{
public:
    Output(size_t n_outputs) : BaseDense(n_outputs) {}

    // Allocate memory for matrices, set the activation and error function
    void build(shape2D input_shape, activations act, error_funcs err = mean_squared_error)
    {
        BaseDense::build(input_shape, act);
        m_error.reset(get_error_function(err));
    }

    void gradient(matrix_t& inputs, matrix_t& expected) 
    {
        auto dA = m_func_deriv(m_activations);
        auto dC = m_error->d_cost(m_activations, expected);


        // std::cout << "dA: " << dA << '\n';
        // std::cout << "dC: " << dC << '\n';

        m_partial_deriv = dA % dC;

        // std::cout << "P: " << m_partial_deriv << '\n';

        m_d_weights     = inputs.T() * m_partial_deriv;

        // std::cout << "dW: " << m_d_weights << '\n';

        m_d_biases      = m_partial_deriv;

        // std::cout << "dB: " << m_d_biases << '\n';
    }

    number_t cost(matrix_t& expected)
    {
        return m_error->error(m_activations, expected);
    }

private:
    base_error_func_t m_error;
};

END_BLUST_NAMESPACE;