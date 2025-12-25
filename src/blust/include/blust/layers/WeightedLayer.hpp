#pragma once

#include <blust/layers/BaseWeightedLayer.hpp>

START_BLUST_NAMESPACE

// Fully connected layer, with weights and partial derivatives
class WeightedLayer : public BaseWeightedLayer
{
protected:

    tensor_t m_weights;
    tensor_t m_transp_weights;
    tensor_t m_d_weights;
    tensor_t m_weighted_input;
    tensor_t m_partial_deriv;
public:

    friend class nn_ops;
    friend class Dense;

    WeightedLayer() = default;
    WeightedLayer(const WeightedLayer& other) : BaseWeightedLayer(other)
    {
        m_weights           = other.m_weights;
        m_d_weights         = other.m_d_weights;
        m_weighted_input    = other.m_weighted_input;
        m_partial_deriv     = other.m_partial_deriv;
        m_transp_weights   = other.m_transp_weights;
    }

    WeightedLayer(WeightedLayer&& other) : BaseWeightedLayer(std::forward<BaseWeightedLayer>(other))
    {
        m_weights           = std::move(other.m_weights);
        m_d_weights         = std::move(other.m_d_weights);
        m_weighted_input    = std::move(other.m_weighted_input);
        m_partial_deriv     = std::move(other.m_partial_deriv);
        m_transp_weights   = std::move(other.m_transp_weights);
    }

    tensor_t& get_weights() override { return m_weights; }
    tensor_t& get_partial_deriv() override { return m_partial_deriv; }
    tensor_t& get_weighted_input() override { return m_weighted_input; }
    tensor_t& get_gradient_w() override { return m_d_weights; }
    tensor_t& get_transp_weights() { return m_transp_weights; }

    virtual size_t bytesize() const override {
        return BaseWeightedLayer::bytesize() + m_weights.bytesize() + m_d_weights.bytesize() +
               m_weighted_input.bytesize() + m_partial_deriv.bytesize() + m_transp_weights.bytesize();
    }
};


END_BLUST_NAMESPACE