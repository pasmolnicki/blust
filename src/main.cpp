#include <blust/blust.hpp>
#include <chrono>
#include <iostream>

size_t m = 7, n = 7, k = 7;

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

void tensor_mul_test() {
	size_t bytes_size;

	tensor t1({m, n}, 2);
	tensor t2({m, n}, 2);
	tensor t3({n, k}, 2);
	tensor t4({n, k}, 2);

	bytes_size = t1.bytesize() + t2.bytesize();

	using namespace std::chrono;

	// for each result Cij = Dot(Aix, Bxj) x from 0 to m
	// dot has 2 operations: +, * so we get total of n * k * [Dot(Aix, Bxj) x from 0 to m] operations
	// and that is n * k * (2m) = 2nkm
	const double FLOPS = 2 * n * m * k + n * m + m * k;
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
		t3.fill([&dist, &gen](){ return dist(gen); });
		t4.fill([&dist, &gen](){ return dist(gen); });
	}

	std::cout.setf(std::ios::fixed);
	std::cout.precision(3);
	std::cout << "Size: " << bytes_size / 1e6 << "MB\n";
	std::cout << "Starting...\n";

	cpu_ops cops(std::thread::hardware_concurrency());

	constexpr size_t n_iter = 5;
	// for (auto mc : {64, 96, 128, 192, 256}) {
	// 	for (auto kc : {96, 128, 192, 256, 384, 512, 1024}) {
	// 		for (auto nc : {512, 1024, 2048, 4096}) {
	// 			// Warm up
	// 			r = cops.mat_mul(cops.add(t1, t2), cops.add(t3, t4), mc, kc, nc);

	// 			auto start = high_resolution_clock::now();
	// 			for (i = 0; i < n_iter; i++)
	// 				r = cops.mat_mul(cops.add(t1, t2), cops.add(t3, t4), mc, kc, nc);
	// 			seconds = duration_cast<microseconds>(high_resolution_clock::now() - start).count() / 1e6 / n_iter;
	// 			auto gflops  = FLOPS / (seconds) / 1e9;

	// 			std::cout 
	// 				<< "MC=" << mc << " KC=" << kc << " NC=" << nc
	// 				<< " time=" << seconds
	// 				<< "s gflops="<< gflops 
	// 				<< "\n";
	// 		}
	// 	}
	// }

	r = cops.mat_mul(cops.add(t1, t2), cops.add(t3, t4), 256, 128, 256);
	auto start = high_resolution_clock::now();
	for (i = 0; i < n_iter; i++)
		r = cops.mat_mul(cops.add(t1, t2), cops.add(t3, t4));

	seconds = duration_cast<microseconds>(high_resolution_clock::now() - start).count() / 1e6 / n_iter;
	auto gflops  = FLOPS / (seconds) / 1e9;

	if (n < MAX_PRINT_DIM && m < MAX_PRINT_DIM && k < MAX_PRINT_DIM)
		std::cout << r << '\n';

	std::cout 
		<< "time=" << seconds
			<< "s gflops="<< gflops 
			<< " n_allocs=" << utils::n_allocs
			<< " max_allocs=" << utils::max_allocs
			<< " n_shared=" << utils::n_shared
			<< " max_shared=" << utils::max_shared
			<< "\n";
	
	std::cout << "Testing result...\n";


	auto T1 = cops.add(t1, t2), T2 = cops.add(t3, t4);
	auto naiveSeconds = test_mat_mul(T1.data(), T2.data(), r.data());
	std::cout << "Naive time: " << naiveSeconds << "s\n";
	std::cout << "Speedup: " << naiveSeconds / seconds << "x\n";

	std::cout << "Press enter...\n";
	
	getchar();
}

void tesor_add_test() {
	size_t bytes_size;

	tensor t1({m * m});
	tensor t2({m * m});
	tensor t3({m * m});
	tensor t4({m * m});

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
		t3.fill([&dist, &gen](){ return dist(gen); });
		t4.fill([&dist, &gen](){ return dist(gen); });
	}

	std::cout.setf(std::ios::fixed);
	std::cout.precision(3);
	std::cout << "Size: " << bytes_size / 1e6 << "MB\n";
	std::cout << "Starting...\n";

	auto start = high_resolution_clock::now();
	constexpr size_t n_iter = 5;
	for (i = 0; i < n_iter; i++)
		r = ops->add(ops->add(t1, t2), ops->add(t3, t4));

	seconds = duration_cast<microseconds>(high_resolution_clock::now() - start).count() / 1e6 / n_iter;
	gflops  = gflops / (seconds) / 1e9;

	if (n < MAX_PRINT_DIM && m < MAX_PRINT_DIM && k < MAX_PRINT_DIM)
		std::cout << r << '\n';

	std::cout 
		<< "time=" << seconds
			<< "s gflops="<< gflops 
			<< " n_allocs=" << utils::n_allocs
			<< " max_allocs=" << utils::max_allocs
			<< " n_shared=" << utils::n_shared
			<< " max_shared=" << utils::max_shared
			<< "\n";
	
	std::cout << "Testing result...\n";

	auto T1 = ops->add(t1, t2), T2 = ops->add(t3, t4);
	auto naiveSeconds = test_result(T1.data(), T2.data(), r.data(), r.size());
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

	tensor_t expected{ {1, 2} };
	expected(0) = 1.0f;
	expected(1) = 0.0f;
	batch_t batch_input = { inputs };
	batch_t batch_expected = { expected };

	auto start = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < 10; i++)
		seq.train_on_batch(batch_input, batch_expected);
	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> duration = end - start;
	printf("avg time: %f ms\n", duration.count() / 10.0f);
}

void mnistTest() {
	// Load MNIST dataset
	auto [train_images, train_labels] = blust::mnist::load();

	Sequential seq;	
	seq.add(Input({ 1, 784 }));
	seq.add(Dense(128, relu));
	seq.add(Dense(64, relu));
	seq.add(Dense(10, softmax));

	seq.compile(new SGD(0.9), error_funcs::mean_squared_error);

	auto error_before = MeanSquaredError().error(
		seq.predict(train_images[0]), train_labels[0]);

	auto start = std::chrono::high_resolution_clock::now();
	seq.fit(train_images, train_labels, 64);
	auto end = std::chrono::high_resolution_clock::now();

	std::cout << std::format("Training completed in {:.2f} seconds\n",
		std::chrono::duration<double>(end - start).count());
}

void test_mat_mul_opencl() {
	opencl_ops ops;

	// m = 1024; n = 1024; k = 1024;
	
	tensor_t a({ m, n }, 1.0f, tensor_t::pointer_type::opencl);
	tensor_t b({ n, k }, 4.0f, tensor_t::pointer_type::opencl);
	tensor_t c;

	c = ops.mat_mul(a, b); // Warm up

	auto start = std::chrono::high_resolution_clock::now();
	c = ops.mat_mul(a, b);
	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> duration = end - start;
	printf("OpenCL MatMul time: %.3fms GFLOPS=%.2f\n", duration.count(), 
		(2.0 * m * n * k) / (duration.count() / 1e3) / 1e9);

	cpu_ops cops;
	a.to_host();
	b.to_host();
	start = std::chrono::high_resolution_clock::now();
	auto r = cops.mat_mul(a, b);
	end = std::chrono::high_resolution_clock::now();
	duration = end - start;
	printf("CPU MatMul time: %.3fms GFLOPS=%.2f\n", duration.count(), 
		(2.0 * m * n * k) / (duration.count() / 1e3) / 1e9);

	// Test
	c.to_host();
	auto c_data = c.data();
	auto r_data = r.data();
	bool ok = true;
	for (size_t i = 0; i < c.size(); i++) {
		if (fabs(c_data[i] - r_data[i]) > 1e-2) {
			std::cout << i << ": " << c_data[i] << " != " << r_data[i] << "\n";
			ok = false;
			break;
		}
	}

	if (n < MAX_PRINT_DIM && m < MAX_PRINT_DIM && k < MAX_PRINT_DIM) {
		// A,B
		std::cout << "A:\n" << a << std::endl;
		std::cout << "B:\n" << b << std::endl;

		// OpenCL
		std::cout << "OpenCL:\n" << c << std::endl;
		// Cpu
		std::cout << "CPU:\n" << r << std::endl;
	}

	auto naive_time = test_mat_mul(a.data(), b.data(), c.data());
	printf("Naive MatMul time: %.3fms GFLOPS=%.2f\n", naive_time * 1e3, 
		(2.0 * m * n * k) / (naive_time) / 1e9);

	if (ok) {
		std::cout << "OpenCL matmul test passed!\n";
	} else {
		std::cout << "OpenCL matmul test failed!\n";
	}
}

void test_opencl() {
	opencl_ops ops;
	
	tensor_t a({ 1024 * 1024 }, 1.0f, tensor_t::pointer_type::opencl);
	tensor_t b({ 1024 * 1024 }, 4.0f, tensor_t::pointer_type::opencl);
	tensor_t c;

	auto start = std::chrono::high_resolution_clock::now();
	c = ops.add(a, b);
	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> duration = end - start;
	printf("OpenCL Add time: %.3f ms\n", duration.count());

	cpu_ops cops;
	a.to_host();
	b.to_host();
	start = std::chrono::high_resolution_clock::now();
	cops.add(a, b);
	end = std::chrono::high_resolution_clock::now();
	duration = end - start;
	printf("CPU Add time: %.3f ms\n", duration.count());

	// Test
	c.to_host();
	auto c_data = c.data();
	bool ok = true;
	for (size_t i = 0; i < c.size(); i++) {
		if (fabs(c_data[i] - 5.0f) > 1e-2) {
			std::cout << i << ": " << c_data[i] << " != 3.0\n";
			ok = false;
			break;
		}
	}
	if (ok) {
		std::cout << "OpenCL add test passed!\n";
	} else {
		std::cout << "OpenCL add test failed!\n";
	}
}

void test_numpy_opencl() {
	opencl_ops ops;

	m = 784; k = 4096; n = 512;

	tensor_t t1({m, k}, 0.0f, tensor_t::pointer_type::opencl);
	tensor_t t2({m, k}, 0.0f, tensor_t::pointer_type::opencl);
	tensor_t t3({k, n}, 0.0f, tensor_t::pointer_type::opencl);
	tensor_t t4({k, n}, 0.0f, tensor_t::pointer_type::opencl);

	utils::randomize(t1);
	utils::randomize(t2);
	utils::randomize(t3);
	utils::randomize(t4);

	tensor_t r;

	// Warm up
	for (int i = 0; i < 10; i++)
		r = ops.mat_mul(ops.add(t1, t2), ops.add(t3, t4));

	auto start = std::chrono::high_resolution_clock::now();
	constexpr size_t n_iter = 5;
	for (size_t i = 0; i < n_iter; i++)
		r = ops.mat_mul(t1, t3);
	auto end = std::chrono::high_resolution_clock::now();

	// Test against cpu
	cpu_ops cops;
	tensor_t T1 
		// = cops.add(t1, t2);
		= t1;
	tensor_t T2 
		// = cops.add(t3, t4);
		= t3;
	auto c_r = cops.mat_mul(T1, T2);

	for (size_t i = 0; i < r.size(); i++) {
		if (fabs(r.data()[i] - c_r.data()[i]) > 1e-2) {
			std::cout << i << ": " << r.data()[i] << " != " << c_r.data()[i] << "\n";
			printf("OpenCL MatMul test failed!\n");
			return;
		}
	}

	printf("OpenCL MatMul test passed!\n");
	std::chrono::duration<double, std::milli> duration = end - start;
	printf("OpenCL MatMul time: %.3f ms GFLOPS=%.2f\n", duration.count() / n_iter, 
		(2.0 * m * n * k) / ((duration.count() / n_iter) / 1e3) / 1e9);
}

void perf_mat_mul(tensor_t& a, tensor_t& b) {
	tensor_t r;
	r = ops->mat_mul(a, b);
}

int main(int argc, char** argv)
{
    init(argc, argv, "cpu");

	printf("Tensor:\n");

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

	// tensor_t t1({m, n});
	// tensor_t t2({n, k});
	// utils::randomize(t1.begin(), t1.end(), t1.size());
	// utils::randomize(t2.begin(), t2.end(), t2.size());
	// perf_mat_mul(t1, t2);
	// std::cout 
			// << " n_allocs=" << utils::n_allocs
			// << " max_allocs=" << utils::max_allocs
			// << " n_shared=" << utils::n_shared
			// << " max_shared=" << utils::max_shared
			// << "\n";
	std::cout << g_settings->backend() << "\n";

	// test_opencl();
	// test_numpy_opencl();
	// test_mat_mul_opencl();
	// tensor_mul_test();
	// tesor_add_test();
	// modelTest();
	mnistTest();

	return 0;
}
