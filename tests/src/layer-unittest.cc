#include <gtest/gtest.h>

#include <blust/blust.hpp>
TEST(LayerTest, TestBuildingLayers)
{
    using namespace blust;

    Output layer(5);
    matrix_t inputs{{1.0f, .1f}};
    layer.build(inputs.dim(), softmax);

    // Assert correct dimensions of the input, layer weights and biases
    ASSERT_EQ(shape2D(1, 2), inputs.dim());
    ASSERT_EQ(shape2D(2, 5), layer.get_weights().dim());
    ASSERT_EQ(shape2D(1, 5), layer.get_biases().dim());
    ASSERT_EQ(shape2D(1, 5), layer.dim());
}

TEST(LayerTest, TestBasicNetworkLearning)
{
    using namespace blust;

    // Setup basic network
    Output layer(3);
    Dense hidden(4);

    // inputs
    matrix_t inputs({{1.0f, 0.1f, .5f}});

    // Build the network, starting from hidden to output layer
    hidden.build(inputs.dim(), relu);
    layer.build(hidden.dim(), softmax);
    
    // Set random value for weights and biases
    hidden.randomize();
    layer.randomize();

    // Feed forward the network
    auto hidden_out = hidden.feed_forward(inputs);
    auto output     = layer.feed_forward(hidden_out);

    // Calculate the cost
    matrix_t expected{{1, 0, 0}};
    auto cost1 = layer.cost(expected);

    // Calculate the gradients
    layer.gradient(hidden_out, expected);
    hidden.gradient(&layer, inputs);

    // Apply them
    layer.apply(0.5);
    hidden.apply(0.5);

    // Feed forward again, and check the cost
    hidden_out = hidden.feed_forward(inputs);
    output     = layer.feed_forward(hidden_out);

    // Network should learn, minimizing the cost
    // cost1 > current cost
    ASSERT_GT(cost1, layer.cost(expected));
}