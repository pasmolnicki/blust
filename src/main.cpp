#include <blust/blust.hpp>

using namespace blust;

int main()
{
    Input input     = Input({1, 2});
    Dense hidden    = Dense(4, relu)(input);
    Dense feature   = Dense(2, softmax)(hidden);

    matrix_t inputs({{1.0f, 0.0f}});
    
    hidden.randomize();
    feature.randomize();

    Model model(&input, &feature);
    model.compile(0.2);

    matrix_t expected{{0, 1}};
    std::vector<matrix_t> batch_input = {inputs};
    std::vector<matrix_t> batch_expected = {expected};

    model.compile(0.8);
    model.train_on_batch(batch_input, batch_expected);
    model.train_on_batch(batch_input, batch_expected);

    return 0;
}