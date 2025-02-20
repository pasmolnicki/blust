#include <blust/blust.hpp>

using namespace blust;

int main()
{
    Output layer(2);

    matrix_t inputs({{1.0f, 0.1f, .5f}});

    layer.build(inputs.dim(), softmax);
    layer.randomize(0x244);

    std::cout << "I: " << inputs << '\n';
    std::cout << "W: " <<layer.get_weights() << '\n';
    std::cout << "B: " <<layer.get_biases() << '\n';

    auto& outputs = layer.feed_forward(inputs);

    std::cout << "WI:" << layer.get_weighted_input() << '\n';

    std::cout << "O: " << outputs << '\n';

    matrix_t expected({{0, 1}});
    std::cout << "cost=" << layer.cost(expected) << '\n';

    layer.gradient(inputs, expected);
    layer.apply(1.2);

    outputs = layer.feed_forward(inputs);
    std::cout << "cost=" << layer.cost(expected) << '\n';

    return 0;
}