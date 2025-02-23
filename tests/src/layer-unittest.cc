#include <gtest/gtest.h>

#include <blust/blust.hpp>
TEST(LayerTest, TestBuildingLayers)
{
    using namespace blust;

    Dense layer(5);
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

    Dense hidden(4);
    Dense feature(3);

    matrix_t inputs({{1.0f, 0.1f, .5f}});

    hidden.build(inputs.dim(), relu);
    feature.build(hidden.dim(), softmax);
    feature.attach(&hidden);
    
    hidden.randomize();
    feature.randomize();

    auto hidden_out = hidden.feed_forward(inputs);
    auto output     = feature.feed_forward(hidden_out);

    matrix_t expected{{1, 0, 0}};

    auto cost1 = feature.cost(expected);

    feature.gradient(hidden_out, expected);
    hidden.gradient(inputs);

    // layer.apply(0.2);
    hidden.apply(10);

    hidden_out = hidden.feed_forward(inputs);
    output     = feature.feed_forward(hidden_out);

    // Network should learn, minimizing the cost
    // cost1 > current cost
    ASSERT_GT(cost1, feature.cost(expected));
}