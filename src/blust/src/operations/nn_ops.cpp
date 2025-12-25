#include <blust/backend/nn_ops.hpp>
#include <blust/layers/Dense.hpp>

START_BLUST_NAMESPACE

void nn_ops::nn_hidden_gradient(Dense* layer)
{
    // Calculate the partial derivative:
    // N_(L-1) = P_(L) * W_(L).T
    // P_(L-1) = N_(L-1) % A_(L-1)
    // then dC/dW_(L-1) = A_(L-2).T * P_(L-1)
    auto next       = dynamic_cast<WeightedLayer*>(layer->m_next);
    auto N          = ops->mat_mul(next->m_partial_deriv, next->m_transp_weights);
    
    layer->m_func_activ->derivative(layer->m_activations, layer->m_partial_deriv);
    ops->hadamard(N, layer->m_partial_deriv, layer->m_partial_deriv);

    auto w_grad = ops->mat_mul(layer->m_prev->m_transp_activations, layer->m_partial_deriv);

    // Accumulate gradients
    ops->add(layer->m_d_weights, w_grad, layer->m_d_weights);
    ops->add(layer->m_d_biases, layer->m_partial_deriv, layer->m_d_biases);
}

void nn_ops::nn_output_gradient(Dense* layer, tensor_t& expected, error_functor_t& func)
{
    // dC/dW_L = A_(L-1).T * (dA % dC)
    layer->m_func_activ->derivative(layer->m_activations, layer->m_partial_deriv);
    auto dC          = func->d_cost(layer->m_activations, expected);

    ops->hadamard(layer->m_partial_deriv, dC, layer->m_partial_deriv);
    auto w_grad      = ops->mat_mul(layer->m_prev->m_transp_activations, layer->m_partial_deriv);

    // Accumulate gradients
    ops->add(layer->m_d_weights, w_grad, layer->m_d_weights);
    ops->add(layer->m_d_biases, layer->m_partial_deriv, layer->m_d_biases);
}

void nn_ops::nn_feed_forward(Dense* layer, tensor_t& inputs)
{
    // m_weighted_input = (1) inputs * m_weights + (2) m_biases
    layer->m_weighted_input.fill(0); // Matmul will simply add to the existing buffer
    ops->mat_mul(inputs, layer->m_weights, layer->m_weighted_input);
    ops->add(layer->m_biases, layer->m_weighted_input, layer->m_weighted_input);
    layer->m_func_activ->activation(layer->m_weighted_input, layer->m_activations);
}

END_BLUST_NAMESPACE