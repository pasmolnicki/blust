#include <blust/blust.hpp>
#include <chrono>
#include <iostream>
#include <sys/time.h>

using namespace blust;

void test_result(tensor& a, tensor& b, tensor& c) {

	auto a_data = a.data();
	auto b_data = b.data();
	auto c_data = c.data();
	auto size = a.size ();

	for (size_t i = 0; i < size; i++) {
		if (fabs(c_data[i] - (a_data[i] * b_data[i])) > 1e-6 ) {
			std::cout << i << ": " << a_data[i] << " + " << b_data[i] << " != " << c_data[i] << std::endl;
			return;
		}
	}

	std::cout << "Test passed!\n";
}

int main(int argc, char** argv)
{
    init(argc, argv, "");
	// batch_t dataset, labels;
	// mnist::load_dataset(dataset, labels);

	// Sequential seq;
	// seq.add(Input({ 1, 784 }));
	// seq.add(Dense(512, relu));
	// seq.add(Dense(64, sigmoid));
	// seq.add(Dense(10, softmax));

	// seq.compile(new SGD(1e-2, 0.9, true));
	// seq.fit(dataset, labels, 30);

	int n = 10000, m = 20000;

	tensor t({n, m}, 1.25);
	tensor t1({n, m}, 2.5);


	// std::cout << t << '\n';
	// std::cout << t1 << '\n';
	tensor r = ops->hadamard(t, t1);	
	test_result(t1, t, r);

	// std::cout << r << '\n';

	return 0;
}
	/*Sequential seq;
	({
		new Input({1, 768}),
		new Dense(2048, relu),
		new Dense(512, relu),
		new Dense(128, relu),
		new Dense(512, relu),
		new Dense(64, relu),
		new Dense(2, softmax)
		});
	seq.add(Input({ 1, 768 }));
	seq.add(Dense(2048, relu));
	seq.add(Dense(512, relu));
	seq.add(Dense(128, relu));
	seq.add(Dense(512, relu));
	seq.add(Dense(64, relu));
	seq.add(Dense(2, softmax));

	seq.compile(0.1);

	matrix_t inputs({ 1, 768 }, 0.5f);
	utils::randomize(inputs.begin(), inputs.end(), inputs.size());

	matrix_t expected{ {0, 1} };
	batch_t batch_input = { inputs };
	batch_t batch_expected = { expected };

	auto start = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < 10; i++)
		seq.train_on_batch(batch_input, batch_expected);
	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> duration = end - start;
	printf("avg time: %f ms\n", duration.count() / 10.0f);*/

	/*Input input = Input({1, 768});
    Dense hidden    = Dense(2048, relu)(input);
    Dense hidden2   = Dense(512, relu)(hidden);
    Dense hidden3   = Dense(128, relu)(hidden2);
    Dense hidden4   = Dense(512, relu)(hidden3);
    Dense hidden5   = Dense(64, relu)(hidden4);
    Dense feature   = Dense(2, softmax)(hidden5);
    matrix_t inputs({1, 768 }, 0.5f);

	utils::randomize(inputs.begin(), inputs.end(), inputs.size());

    hidden.randomize();
    feature.randomize();

    Model model(&input, &feature);
    model.compile(0.1);

    matrix_t expected{{0, 1}};
    batch_t batch_input = {inputs};
    batch_t batch_expected = {expected};

	auto start = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < 10; i++)
		model.train_on_batch(batch_input, batch_expected);
	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> duration = end - start;
	printf("avg time: %f ms\n", duration.count() / 10.0f);

    return 0;*/
