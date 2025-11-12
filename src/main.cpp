#include <blust/blust.hpp>
#include <chrono>
#include <iostream>

size_t m = 1024, n = 1024, k = 1024;

using namespace blust;

double test_result(number_t* a_data, number_t* b_data, number_t* c_data, size_t size) {
	using namespace std::chrono;
	auto start = steady_clock::now();
	for (size_t i = 0; i < size; i++) {
		if (fabs(c_data[i] - (a_data[i] + b_data[i])) > 1e-2 ) {
			std::cout << i << ": " << a_data[i] << " + " << b_data[i] << " != " << c_data[i] << std::endl;
			return -1;
		}
	}
	auto total = duration_cast<microseconds>(steady_clock::now() - start).count() / 1e6;


	std::cout << "Test passed!\n";

	return total;
}

constexpr auto MAX_PRINT_DIM = 20;

double test_mat_mul(number_t* a_data, number_t* b_data, number_t* c_data)
{
	using namespace std::chrono;
	tensor r{{m, k}};
	auto r_data = r.data();
	
	auto start = steady_clock::now();

	for(int row = 0; row < m; row++)
	{
		for(int column = 0; column < k; column++) 
		{       
			r_data[row * k + column] = 0; 
			for(int element = 0; element < n; element++)  
			{
				r_data[row * k + column] += a_data[row * n + element] * b_data[element * k + column]; 
			}
		}
	}
	auto total = duration_cast<microseconds>(steady_clock::now() - start).count() / 1e6;

	if (n < MAX_PRINT_DIM && m < MAX_PRINT_DIM && k < MAX_PRINT_DIM)
		std::cout << r << std::endl;

	auto size = r.size();
	for (size_t i = 0; i < size; i++) {
		if (fabs(r_data[i] - c_data[i]) > 1e-2 ) {

			std::cout 
					<< "(" << i / k << ", " << i % k << ") (test != result) " 
					<< r_data[i] << " != " << c_data[i] << '\n';
			return total;
		}
	}

	std::cout << "test passed!\n";

	return total;
}

void tensor_mul_test(int argc, char** argv) {
	if (argc == 4) {
		std::cout << "Custom args\n";
		int a[3] = {};
		bool ok = true;
		for (int i = 0; i < 3; i++) {
			int scanned = sscanf(argv[i+1], "%d", &a[i]);
			if (scanned == 0) {
				std::cout << argv[i+1] << " is not a valid dim\n";
				ok = false;
				break;
			}
		}

		if (ok) {
			m = a[0];
			n = a[1];
			k = a[2];

			std::cout << "Using m=" << m 
					  << " n=" << n 
					  << " k=" << k << '\n';
		}
	}

	size_t bytes_size;

	tensor t1({m, n}, 2);
	tensor t2({n, k}, 2);

	bytes_size = t1.bytesize() + t2.bytesize();

	using namespace std::chrono;

	// for each result Cij = Dot(Aix, Bxj) x from 0 to m
	// dot has 2 operations: +, * so we get total of n * k * [Dot(Aix, Bxj) x from 0 to m] operations
	// and that is n * k * (2m) = 2nkm
	double gflops = 2 * n * m * k;
	double seconds;
	
	tensor r;
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

	std::cout.setf(std::ios::fixed);
	std::cout.precision(3);
	std::cout << "Size: " << bytes_size / 1e6 << "MB\n";
	std::cout << "Starting...\n";

	auto start = high_resolution_clock::now();
	constexpr size_t n_iter = 5;
	for (i = 0; i < n_iter; i++)
		r = ops->mat_mul(t1, t2);

	seconds = duration_cast<microseconds>(high_resolution_clock::now() - start).count() / 1e6 / n_iter;
	gflops  = gflops / (seconds) / 1e9;

	if (n < MAX_PRINT_DIM && m < MAX_PRINT_DIM && k < MAX_PRINT_DIM)
		std::cout << r << '\n';

	std::cout 
		<< "time=" << seconds
			<< "s gflops="<< gflops 
			<< " n_allocs=" << tensor::n_allocs 
			<< " max_allocs=" << tensor::max_allocs
			<< "\n";
	
	std::cout << "Testing result...\n";

	auto naiveSeconds = test_mat_mul(t1.data(), t2.data(), r.data());
	std::cout << "Naive time: " << naiveSeconds << "s\n";
	std::cout << "Speedup: " << naiveSeconds / seconds << "x\n";

	std::cout << "Press enter...\n";
	
	getchar();
}

void tesor_add_test() {
	size_t bytes_size;

	tensor t1({m * m});
	tensor t2({m * m});

	bytes_size = t1.bytesize() + t2.bytesize();

	using namespace std::chrono;
	double gflops = 2 * m * m;
	double seconds;
	
	tensor r;
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

	std::cout.setf(std::ios::fixed);
	std::cout.precision(3);
	std::cout << "Size: " << bytes_size / 1e6 << "MB\n";
	std::cout << "Starting...\n";

	auto start = high_resolution_clock::now();
	constexpr size_t n_iter = 5;
	for (i = 0; i < n_iter; i++)
		r = ops->add(t1, t2);

	seconds = duration_cast<microseconds>(high_resolution_clock::now() - start).count() / 1e6 / n_iter;
	gflops  = gflops / (seconds) / 1e9;

	if (n < MAX_PRINT_DIM && m < MAX_PRINT_DIM && k < MAX_PRINT_DIM)
		std::cout << r << '\n';

	std::cout 
		<< "time=" << seconds
			<< "s gflops="<< gflops 
			<< " n_allocs=" << tensor::n_allocs 
			<< " max_allocs=" << tensor::max_allocs
			<< "\n";
	
	std::cout << "Testing result...\n";

	auto naiveSeconds = test_result(t1.data(), t2.data(), r.data(), r.size());
	std::cout << "Naive time: " << naiveSeconds << "s\n";
	std::cout << "Speedup: " << naiveSeconds / seconds << "x\n";

	std::cout << "Press enter...\n";
	
	getchar();
}

void modelTest() {
	Sequential seq;	
	seq.add(Input({ 1, 768 }));
	seq.add(Dense(512, relu));
	seq.add(Dense(64, relu));
	seq.add(Dense(2, softmax));

	seq.compile();

	tensor_t inputs({ 1, 768 }, 0.5f);
	utils::randomize(inputs.begin(), inputs.end(), inputs.size());

	tensor_t expected{ {0, 1} };
	batch_t batch_input = { inputs };
	batch_t batch_expected = { expected };

	auto start = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < 10; i++)
		seq.train_on_batch(batch_input, batch_expected);
	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> duration = end - start;
	printf("avg time: %f ms\n", duration.count() / 10.0f);

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
	printf("avg time: %f ms\n", duration.count() / 10.0f);*/
}

void matrix_mul_test() {
	size_t bytes_size;

	matrix_t m1({n, m});
	matrix_t m2({m, k});
	// matrix_t m3({m, k});

	bytes_size = m1.bytesize() + m2.bytesize();

	using namespace std::chrono;

	double gflops =  2 * n * m * k; 
	double seconds;
	
	matrix_t r;
	number_t i = 1;

	if (n < MAX_PRINT_DIM && m < MAX_PRINT_DIM && k < MAX_PRINT_DIM)
	{
		m1.fill([&i](){ return i++; });
		m2.fill([&i](){ return i++; });
		std::cout << m1 << std::endl;
		std::cout << m2 << std::endl;
	}
	else
	{
		std::random_device rd{};
		std::mt19937 gen{rd()};
		std::uniform_real_distribution<number_t> dist{0, 1};
		m1.fill([&dist, &gen](){ return dist(gen); });
		m2.fill([&dist, &gen](){ return dist(gen); });
	}

	std::cout << "Size: " << bytes_size / 1e6 << "MB\n";
	std::cout << "Starting...\n";

	auto start = high_resolution_clock::now();
	constexpr size_t n_iter = 25;
	for (i = 0; i < n_iter; i++)
		// r = ops->add(t1, t2);
		r = m1 * m2;

	// ops->add(t1, t2);
	// r = ops->add(ops->mat_mul(t1, t2), t);

	seconds = duration_cast<microseconds>(high_resolution_clock::now() - start).count() / 1e6 / 25;
	gflops  = gflops / (seconds) / 1e9;

	if (n < MAX_PRINT_DIM && m < MAX_PRINT_DIM && k < MAX_PRINT_DIM)
		std::cout << r << '\n';

	std::cout 
		<< "time=" << seconds
			<< "s gflops="<< gflops 
			<< " n_allocs=" << tensor::n_allocs 
			<< " max_allocs=" << tensor::max_allocs
			<< "\n";
	
	std::cout << "Testing result...\n";

	// test_result(m1.data(), m2.data(), r.data(), r.size());
	test_mat_mul(m1.data(), m2.data(), r.data());
	// test_mat_mul(t1, t2, r);

	std::cout << "Press enter...\n";
	
	getchar();
}

int main(int argc, char** argv)
{
    init(argc, argv, "cpu");

	printf("Tensor:\n");
	tensor_mul_test(argc, argv);
	// tesor_add_test();
	// modelTest();
}
