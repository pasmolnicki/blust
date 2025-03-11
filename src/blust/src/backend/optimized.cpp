#include <blust/backend/optimized.hpp>

START_BLUST_NAMESPACE

// Initialize the optimized backend, and set the threshold values
void optimized_backend::M_set_threshold()
{
	m_threshold_vector_size = 0;
	m_threshold_matrix_size = 0;

	// Measure the time of the operation on CPU and GPU
	// and set the threshold when to use the GPU
	if (m_cuda.is_available())
	{
		// Run tests, and set the threshold values
		m_threshold_vector_size = M_get_size_threshold([](
			base_backend* backend, number_t* res, number_t* mat1, number_t* mat2, size_t n)
			{
				backend->vector_add(res, mat1, mat2, n * n);
			});

		m_threshold_matrix_size = M_get_size_threshold([](
			base_backend* backend, number_t* res, number_t* mat1, number_t* mat2, size_t n)
			{
				backend->mat_mul(res, mat1, mat2, n, n, n);
			});
	}
}

static bool compare_results(const vector_t& res1, const vector_t& res2, size_t n)
{
	bool same = true;
	for (size_t j = 0; j < res1.size(); ++j) {
		if (res1[j] - res2[j] > 1e-5) {
			printf("res1[%lu, %lu] = %f, res2 = %f\n", j / n, j % n, res1[j], res2[j]);
			same = false;
		}
	}
	return same;
}

// Get the size of the input, when the GPU is faster than the CPU
size_t optimized_backend::M_get_size_threshold(
	fn_backend_t fn
)
{
	using namespace std::chrono;

	Gpu_timer gpu_timer;

	size_t sizes[] = { 64, 96, 128, 256, 512, 768, 1024 };
	const size_t n_sizes = sizeof(sizes) / sizeof(sizes[0]);

	size_t i = 0;
	for (; i < n_sizes; ++i)
	{
		vector_t res1, res2, mat1, mat2;
		size_t N = sizes[i] * sizes[i];
		res1.resize(N, .0f);
		res2.resize(N, .0f);
		mat1.resize(N, 1.0f);
		mat2.resize(N, 2.0f);
		utils::randomize(mat1.begin(), mat1.end(), sizes[i] * sizes[i]);
		utils::randomize(mat2.begin(), mat2.end(), sizes[i] * sizes[i]);

		// Measure the time on GPU
		gpu_timer.start();

		for (int j = 0; j < 100; j++)
			fn(&m_cuda, res1.data(), mat1.data(), mat2.data(), sizes[i]);

		gpu_timer.stop();

		// Measure the time on CPU
		auto cpu_start = high_resolution_clock::now();

		for (int j = 0; j < 100; j++)
			fn(&m_cpu, res2.data(), mat1.data(), mat2.data(), sizes[i]);

		auto cpu_end = high_resolution_clock::now();
		duration<double, std::milli> cpu_duration = (cpu_end - cpu_start);

		// Print the results
		printf("Size: %lu, CPU: %fms, GPU: %fms\n", sizes[i], cpu_duration.count() / 100.0f, gpu_timer.get_time() / 100.0f);

		// Check if the results are the same
		if (!compare_results(res1, res2, sizes[i])) {
			printf("Results are not the same!\n");
			break;
		}

		// Set the threshold
		if (gpu_timer.get_time() < cpu_duration.count()) {
			break;
		}
	}

	i = std::min(i, n_sizes - 1);
	return sizes[i];
}

END_BLUST_NAMESPACE