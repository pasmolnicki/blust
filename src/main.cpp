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
	g_backend = std::make_unique<optimized_backend>(argc, argv);

 //   cuda_backend cuda(argc, argv);
	//cpu_backend cpu;

 //   matrix_t res, res2, mat1, mat2;

 //   mat1.build({ 1024, 1024 }, 1.0f);
 //   mat2.build(mat1.dim(), 2.0f);
 //   res.build(mat1.dim());
 //   res2.build(mat1.dim());

 //   std::cout << "Matrix size: " << mat1.size() << "\n";

 //   // Mierzenie czasu na GPU
 //   CUevent start, stop;
 //   cuEventCreate(&start, 0);
 //   cuEventCreate(&stop, 0);

 //   cuEventRecord(start, 0);
 //   cuda.vector_add(res.data(), mat1.data(), mat2.data(), res.size());
 //   cuEventRecord(stop, 0);

 //   cuEventSynchronize(stop);
 //   float milliseconds = 0;
 //   cuEventElapsedTime(&milliseconds, start, stop);

 //   std::cout << "GPU execution time: " << milliseconds << " ms\n";

 //   // Mierzenie czasu na CPU
 //   auto cpu_start = std::chrono::high_resolution_clock::now();
	//cpu.vector_add(res2.data(), mat1.data(), mat2.data(), res.size());
 //   auto cpu_end = std::chrono::high_resolution_clock::now();
 //   std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;

 //   std::cout << "CPU execution time: " << cpu_duration.count() << " ms\n";

 //   // Test if that works
 //   size_t i = 0;
 //   for (; i < mat1.size(); i++) {
 //       if (res(i) - (res2(i)) > 1e-6)
 //           break;
 //   }

 //   if (i == mat1.size())
 //       std::cout << "TEST PASSED\n";
 //   else
 //       std::cout << "TEST FAILED\n";

 //   std::cout << "Finish work\n";

    return 0;
}

//int main(int argc, char** argv)
//{
//    /* Input input     = Input({ 1, 2 });
//     Dense hidden    = Dense(4, relu)(input);
//     Dense feature   = Dense(2, softmax)(hidden);
//
//     matrix_t inputs({{1.0f, 0.0f}});
//
//     hidden.randomize();
//     feature.randomize();
//
//     Model model(&input, &feature);
//     model.compile(0.2);
//
//     matrix_t expected{{0, 1}};
//     batch_t batch_input = {inputs};
//     batch_t batch_expected = {expected};
//
//     model.compile(0.8);
//     model.train_on_batch(batch_input, batch_expected);
//     model.train_on_batch(batch_input, batch_expected);*/
//
//
//    cuda_backend cuda(argc, argv);
//
//    matrix_t res, mat1, mat2;
//
//    mat1.build({ 128, 128 }, 1.0f);
//    mat2.build(mat1.dim(), 2.0f);
//    res.build(mat1.dim());
//
//	printf("Matrix size: %llu\n", mat1.size());
//
//	
//	cuda.vector_add(res, mat1, mat2);
//    
//    // Test if that works
//    size_t i = 0;
//    for (; i < mat1.size(); i++) {
//        if (res(i) - (mat1(i) + mat2(i)) > 1e-6)
//            break;
//    }
//
//    if (i == mat1.size())
//        std::cout << "TEST PASSED\n";
//    else
//        std::cout << "TEST FAILED\n";
//
//
//	std::cout << "Finish work\n";
//
//    return 0;
//}