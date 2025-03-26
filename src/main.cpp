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

constexpr auto MAX_PRINT_DIM = 20;

void test_mat_mul(tensor& a, tensor& b, tensor& c)
{
	tensor r{{(int)a.dim()[0], (int)b.dim()[1]}};
	auto a_data = a.data();
	auto b_data = b.data();
	auto c_data = c.data();
	auto r_data = r.data();

	size_t n = a.dim()[0], m = a.dim()[1], k = b.dim()[1];
	
	for (size_t i = 0; i < n; i++)
	{
		for (size_t j = 0; j < k; j++)
		{
			number_t sum = 0;
			for (size_t l = 0; l < m; l++)
			{
				sum += a_data[i * m + l] * b_data[l * k + j];
			}
			r_data[i * k + j] = sum;
		}
	}

	if (n < MAX_PRINT_DIM && m < MAX_PRINT_DIM && k < MAX_PRINT_DIM)
		std::cout << r << std::endl;

	auto size = r.size();
	for (size_t i = 0; i < size; i++) {
		if (fabs(r_data[i] - c_data[i]) > 1e-2 ) {
			std::cout << i << ": " << r_data[i] << " != " << c_data[i] << std::endl;
			return;
		}
	}

	std::cout << "test passed!\n";
}

int main(int argc, char** argv)
{
    init(argc, argv, "");

	// int n = 1 * 1e1, m = 1 * 1e1;
	constexpr int n = 4e2, m = 8e2, k = 5e2;

	// tensor t({n, m}, 2);
	tensor t1({n, m}, 2);
	tensor t2({m, k}, 2);
	// tensor t3({n, m}, 2);

	using namespace std::chrono;

	double gflops = 2 * n * m * k;
	double seconds;
	
	tensor r;
	// for (size_t i = 0; i < 25; i++)
	// // r = ops->add(t, t1);
	// r = ops->add(ops->hadamard(t, t1), ops->hadamard(t2, t3));
	number_t i = 1;

	
	if (n < MAX_PRINT_DIM && m < MAX_PRINT_DIM && k < MAX_PRINT_DIM)
	{
		t1.fill([&i](){ return i++; });
		t2.fill([&i](){ return i++; });
		std::cout << t1 << std::endl;
		std::cout << t2 << std::endl;
	}
	else
	{
		std::random_device rd{};
		std::mt19937 gen{rd()};
		std::uniform_real_distribution<number_t> dist{0, 1};
		t1.fill([&dist, &gen](){ return dist(gen); });
		t2.fill([&dist, &gen](){ return dist(gen); });
	}

	auto start = high_resolution_clock::now();

	constexpr size_t n_iter = 10;
	for (size_t i = 0; i < n_iter; i++)
		r = ops->mat_mul(t1, t2);

	seconds = float(duration_cast<microseconds>(high_resolution_clock::now() - start).count()) / 1e6 / n_iter;
	gflops  = gflops / (seconds) / 1e9;

	if (n < MAX_PRINT_DIM && m < MAX_PRINT_DIM && k < MAX_PRINT_DIM)
		std::cout << r << '\n';

	// test_result(t, t1, r);
	test_mat_mul(t1, t2, r);

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
