#include <blust/blust.hpp>

using namespace blust;

int main()
{
    Output layer(3);
    Dense hidden(4);

    matrix_t inputs({{1.0f, 0.1f, .5f}});

    hidden.build(inputs.dim(), relu);
    layer.build(hidden.dim(), softmax);
    
    hidden.randomize();
    layer.randomize();

    auto hidden_out = hidden.feed_forward(inputs);
    auto output     = layer.feed_forward(hidden_out);

    matrix_t expected{{1, 0, 0}};

    std::cout << "cost=" << layer.cost(expected) << '\n';

    layer.gradient(hidden_out, expected);
    hidden.gradient(&layer, inputs);

    // layer.apply(0.2);
    hidden.apply(10);

    hidden_out = hidden.feed_forward(inputs);
    output     = layer.feed_forward(hidden_out);

    std::cout << "cost=" << layer.cost(expected) << '\n';

    return 0;
}