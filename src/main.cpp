#include <blust/blust.hpp>
#include <chrono>
#include <iostream>

using namespace blust;

void test_result(tensor& a, tensor& b, tensor& c) {

	auto a_data = a.data();
	auto b_data = b.data();
	auto c_data = c.data();
	auto size = a.size();

	for (size_t i = 0; i < size; i++) {
		if (fabs(c_data[i] - (8.0 /*a_data[i] + b_data[i]*/)) > 1e-6 ) {
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

	// 800MB, 4GB

	int n = 1 * 1e4, m = 1 * 1e4;

	tensor t({n, m}, 2);
	tensor t1({n, m}, 2);
	tensor t2({n, m}, 2);
	tensor t3({n, m}, 2);

	using namespace std::chrono;

	double gflops = 4 * n * m;
	double seconds;

	auto start = high_resolution_clock::now();
	
	tensor r;
	for (size_t i = 0; i < 25; i++)
	// r = ops->add(t, t1);
	r = ops->add(ops->hadamard(t, t1), ops->hadamard(t2, t3));
	

	seconds = float(duration_cast<microseconds>(high_resolution_clock::now() - start).count()) / 1e6 / 25;
	gflops  = gflops / (seconds) / 1e9;

	test_result(t, t1, r);

	std::cout 
	<< "time=" << seconds
	 << "s gflops="<< gflops 
	 << " n_allocs=" << tensor::n_allocs 
	 << " max_allocs=" << tensor::max_allocs
	 << "\n";
	std::cout << "Press enter...\n";
	
	std::string ent;
	std::cin >> ent;

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
