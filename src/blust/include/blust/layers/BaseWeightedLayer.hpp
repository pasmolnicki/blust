#pragma once

#include <blust/layers/BaseLearningLayer.hpp>

START_BLUST_NAMESPACE

// Base class for weighted layers, has virtual methods for getting the weights, partial derivatives, and weighted inputs
class BaseWeightedLayer : public BaseLearningLayer
{
protected:
    bool m_initialized_weights = false;
public:

    friend class Model;

    BaseWeightedLayer() = default;
    BaseWeightedLayer(const BaseWeightedLayer& other) : BaseLearningLayer(other) {}
    BaseWeightedLayer(BaseWeightedLayer&& other) : BaseLearningLayer(std::forward<BaseLearningLayer>(other)) {}

    virtual void randomize(uint64_t seed = 0x27) = 0;
    virtual tensor_t& get_weights() = 0;
    virtual tensor_t& get_partial_deriv() = 0;
    virtual tensor_t& get_weighted_input() = 0;
    virtual tensor_t& get_gradient_w() = 0;

    virtual size_t bytesize() const override {
        return BaseLearningLayer::bytesize();
    }
};

END_BLUST_NAMESPACE