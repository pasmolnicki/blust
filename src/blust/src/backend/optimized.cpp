#include <blust/backend/optimized.hpp>

START_BLUST_NAMESPACE


void optimized_backend::M_set_threshold()
{
	// Measure the time of the operation on CPU and GPU
	// and set the threshold when to use the GPU

	size_t sizes[] = { 64, 96, 128, 256, 512, 768, 1024 };
	const size_t n_sizes = sizeof(sizes) / sizeof(sizes[0]);

	// Measure the time of the operation on CPU and GPU
	// and set the threshold when to use the GPU

	using namespace std::chrono;

	size_t i = 0;
	for (; i < n_sizes; ++i)
	{
		Gpu_timer gpu_timer;

		matrix_t res1, res2, mat1, mat2;
		mat1.build({ sizes[i], sizes[i] }, 1.0f);
		mat2.build(mat1.dim(), 2.0f);
		res1.build(mat1.dim());
		res2.build(mat1.dim());

		// Measure the time on GPU
		gpu_timer.start();
		//auto gpu_start = high_resolution_clock::now();

		for (int i = 0; i < 100; i++)
			m_cuda.vector_add(res1.data(), mat1.data(), mat2.data(), res1.size());
		//auto gpu_end = high_resolution_clock::now();
		//duration<double, std::milli> gpu_duration = gpu_end - gpu_start;
		gpu_timer.stop();

		// Measure the time on CPU
		auto cpu_start = high_resolution_clock::now();
		for (int i = 0; i < 100; i++)
			m_cpu.vector_add(res2.data(), mat1.data(), mat2.data(), res2.size());
		auto cpu_end = high_resolution_clock::now();

		duration<double, std::milli> cpu_duration = (cpu_end - cpu_start);

		//printf("Size: %llu, CPU: %fms, GPU: %fms\n", sizes[i], cpu_duration.count(), gpu_duration.count());
		printf("Size: %llu, CPU: %fms, GPU: %fms\n", sizes[i], cpu_duration.count() / 100.0f, gpu_timer.get_time() / 100.0f);

		// Check if the results are the same
		for (size_t j = 0; j < res1.size(); ++j)
		{
			if (res1(j) - res2(j) > 1e-5)
			{
				printf("Results are not the same\n");
				break;
			}
		}

		// Set the threshold
		//if (gpu_duration.count() < cpu_duration.count()) {
		if (gpu_timer.get_time() < cpu_duration.count()) {
			break;
		}
	}

	i = std::min(i, n_sizes - 1);
	m_threshold_vector_size = sizes[i] * sizes[i];
	m_threshold_matrix_size = m_threshold_vector_size;
	printf("Threshold vector size: %llu matrix size: %llu\n", m_threshold_vector_size, m_threshold_matrix_size);
}



END_BLUST_NAMESPACE