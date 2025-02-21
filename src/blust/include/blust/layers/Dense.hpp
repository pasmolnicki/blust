#pragma once

#include "BaseLayer.hpp"

START_BLUST_NAMESPACE

// Has fully connected layer of weight to input
class Dense : public BaseDense
{
public:
    Dense(size_t n_outputs) : BaseDense(n_outputs) {}

    void gradient(BaseDense* prev, matrix_t& inputs)
    {
        // Calculate the partial derivative:
        // N_(L-1) = P_(L) * W_(L).T
        // P_(L-1) = N_(L-1) % A_(L-1)
        // then dC/dW_(L-1) = A_(L-2).T * P_(L-1)

        auto N          = prev->get_partial_deriv() * prev->get_weights().T();
        auto dA         = m_func_deriv(m_activations);
        m_partial_deriv = N % dA;

        m_d_weights     = inputs.T() * m_partial_deriv;
        m_d_biases      = m_partial_deriv;
    }
};

END_BLUST_NAMESPACE