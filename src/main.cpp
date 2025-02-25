#include <blust/blust.hpp>
#include <chrono>
#include <iostream>

using namespace blust;


void vector_add_cpu(matrix_t& res, const matrix_t& mat1, const matrix_t& mat2) {
    for (size_t i = 0; i < mat1.size(); ++i) {
        res(i) = mat1(i) + mat2(i);
    }
}

int main(int argc, char** argv)
{
    init(argc, argv, "");

    /*Input input     = Input({ 1, 100 });
    Dense hidden    = Dense(128, relu)(input);
    Dense feature   = Dense(2, softmax)(hidden);
    matrix_t inputs({1, 100}, 0.5f);

    hidden.randomize();
    feature.randomize();

    Model model(&input, &feature);
    model.compile(0.1);

    matrix_t expected{{0, 1}};
    batch_t batch_input = {inputs};
    batch_t batch_expected = {expected};

    model.compile(0.1);

	auto start = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < 10; i++)
		model.train_on_batch(batch_input, batch_expected);
	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> duration = end - start;
	printf("Time: %f ms\n", duration.count());*/
    return 0;
}