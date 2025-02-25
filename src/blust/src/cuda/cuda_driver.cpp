#include <blust/backend/cuda_driver.hpp>

// define input fatbin file
#ifndef FATBIN_FILE
#define FATBIN_FILE "cuda_kernel64.fatbin"
#endif

START_BLUST_NAMESPACE

cuda_backend::cuda_backend(int argc, char ** argv)
{
	init(argc, argv);
}

cuda_backend::~cuda_backend()
{
	checkCudaErrors(cuCtxDestroy(cuContext));
}

// Initialize the CUDA backend
void cuda_backend::init(int argc, char** argv)
{
    // Init CUDA
    std::cout << "CUDA setup...\n";

    // Initialize
    checkCudaErrors(cuInit(0));
    cuDevice = findCudaDeviceDRV(argc, (const char**)argv);

    // Create context
    checkCudaErrors(cuCtxCreate(&cuContext, 0, cuDevice));

    // Find the modlue (fatbin file)
    std::string module_path;
    std::ostringstream fatbin;

    if (!findFatbinPath(FATBIN_FILE, module_path, argv, fatbin)) {
        std::cout << "fatbin file not found. exiting..\n";
        return;
    }
    else {
        std::cout << "> initCUDA loading module: <" << module_path << ">\n";
    }

    // Empty
    if (!fatbin.str().size()) {
        std::cout << "fatbin file empty. exiting..\n";
        return;
    }

    // Create module from binary file (FATBIN)
    checkCudaErrors(cuModuleLoadData(&cuModule, fatbin.str().c_str()));

    // Load all the functions from the module
    const int n_functions = 6;
    CUfunction* all_functions[n_functions] = {
        &cu_vector_add, &cu_vector_sub, &cu_vector_mul_hadamard,
        &cu_vector_mul_scalar, &cu_mat_transpose, &cu_mat_mul
    };

    const char* function_names[n_functions] = {
        VECTOR_ADD, VECTOR_SUB, VECTOR_MUL_HADAMARD,
        VECTOR_MUL_SCALAR, MAT_TRANSPOSE, MAT_MUL
    };

    for (int i = 0; i < n_functions; i++) {
        checkCudaErrors(cuModuleGetFunction(all_functions[i], cuModule, function_names[i]));
    }

    // Run test
    M_run_test();
    m_available = true;
}

// Assumes that `res`, `mat1` and `mat2` has already preallocated buffers
void cuda_backend::M_lanuch_vector_like_kernel(number_t* res, number_t* mat1, number_t* mat2, size_t N, CUfunction kernel)
{
    M_prepare_cuda(res, N, mat1, N, mat2, N);

    // Grid/Block configuration
    int blocksPerGrid = M_get_blocks_per_grid(N);
    void* args[] = { &deviceData1, &deviceData2, &deviceDataResult, &N };

    // Launch the CUDA kernel
    M_launch_kernel(kernel, blocksPerGrid, args);
    M_clean_up_cuda(res, N);
}

void cuda_backend::vector_scalar_mul(number_t* res, number_t* mat, number_t scalar, size_t N)
{
	M_prepare_cuda(res, N, mat, N);

	auto blocksPerGrid = M_get_blocks_per_grid(N);
	void* args[] = { &deviceData1, &scalar, &deviceDataResult, &N};

    M_launch_kernel(cu_vector_mul_scalar, blocksPerGrid, args);
	M_clean_up_cuda(res, false);
}

void cuda_backend::mat_transpose(number_t* res, number_t* mat, size_t rows, size_t cols)
{
    size_t N = rows * cols;
	M_prepare_cuda(res, N, mat, N);

	auto blocksPerGrid = M_get_blocks_per_grid(N);
	void* args[] = { &deviceData1, &deviceDataResult, &rows, &cols };

    M_launch_kernel(cu_mat_transpose, blocksPerGrid, args);
	M_clean_up_cuda(res, false);
}

void cuda_backend::mat_mul(number_t* res, number_t* mat1, number_t* mat2, size_t rows1, size_t cols2, size_t rows2)
{
	size_t N = rows1 * cols2;
	M_prepare_cuda(res, N, mat1, rows1 * rows2, mat2, rows2 * cols2);

	std::pair<int, int> blockDim = { 16, 16 };
	std::pair<int, int> gridDim = { 8, 8 };

	void* args[] = { &deviceData1, &deviceData2, &deviceDataResult, &rows1, &cols2, &rows2 };

    checkCudaErrors(cuLaunchKernel(cu_mat_mul, gridDim.first, gridDim.second, 1,
        blockDim.first, blockDim.second, 1, 0, NULL, args, NULL));

	M_clean_up_cuda(res, N);
}

// Allocate memory on cuda device (deviceData1, deviceData2, deviceDataResult) and copy from host
// given data1 and data2 to deviceData1, deviceData2
void cuda_backend::M_prepare_cuda(number_t* res, size_t r, number_t* mat1, size_t m1, number_t* mat2, size_t m2)
{
    // Allocate memory on GPU
    //printf("Allocating GPU memory: %llu\n", r * sizeof(number_t));
    checkCudaErrors(cuMemAlloc(&deviceData1, m1 * sizeof(number_t)));
    checkCudaErrors(cuMemAlloc(&deviceData2, m2 * sizeof(number_t)));
    checkCudaErrors(cuMemAlloc(&deviceDataResult, r * sizeof(number_t)));

    // Copy the data
    checkCudaErrors(cuMemcpyHtoD(deviceData1, mat1, m1 * sizeof(number_t)));
    checkCudaErrors(cuMemcpyHtoD(deviceData2, mat2, m2 * sizeof(number_t)));
}

// Overload when only one matrix is needed
void cuda_backend::M_prepare_cuda(number_t* res, size_t r, number_t* mat, size_t m)
{
	// Allocate memory on GPU
	//printf("Allocating GPU memory: %llu\n", r * sizeof(number_t));
	checkCudaErrors(cuMemAlloc(&deviceData1, m * sizeof(number_t)));
	checkCudaErrors(cuMemAlloc(&deviceDataResult, r * sizeof(number_t)));

	// Copy the data
	checkCudaErrors(cuMemcpyHtoD(deviceData1, mat, m * sizeof(number_t)));
}

// Copies the device result to `res` and frees the GPU memory
void cuda_backend::M_clean_up_cuda(number_t* res, size_t r, bool all)
{
    // Copy from device to result
    checkCudaErrors(cuMemcpyDtoH(res, deviceDataResult, r * sizeof(number_t)));

    // Free data
    checkCudaErrors(cuMemFree(deviceData1));
    checkCudaErrors(cuMemFree(deviceDataResult));

    if (all)
        checkCudaErrors(cuMemFree(deviceData2));
}

// Allocates an array with random float entries.
void RandomInit(float* data, int n) {
    for (int i = 0; i < n; ++i) {
        data[i] = rand() / (float)RAND_MAX;
    }
}

// Run test 
void cuda_backend::M_run_test() {
    printf("TEST: Vector Addition (Driver API)\n");
    size_t N = 50000;
    size_t size = N * sizeof(float);

    // Allocate input vectors h_A and h_B in host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Initialize input vectors
    RandomInit(h_A, N);
    RandomInit(h_B, N);

    // Allocate vectors in device memory
    checkCudaErrors(cuMemAlloc(&deviceData1, size));

    checkCudaErrors(cuMemAlloc(&deviceData2, size));

    checkCudaErrors(cuMemAlloc(&deviceDataResult, size));

    // Copy vectors from host memory to device memory
    checkCudaErrors(cuMemcpyHtoD(deviceData1, h_A, size));

    checkCudaErrors(cuMemcpyHtoD(deviceData2, h_B, size));

    
    // This is the new CUDA 4.0 API for Kernel Parameter Passing and Kernel
    // Launch (simpler method)

    // Grid/Block configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    //int blocksPerGrid = 32;

    void* args[] = { &deviceData1, &deviceData2, &deviceDataResult, &N };

    // Launch the CUDA kernel
    checkCudaErrors(cuLaunchKernel(cu_vector_add, blocksPerGrid, 1, 1,
        threadsPerBlock, 1, 1, 0, NULL, args, NULL));
    

#ifdef _DEBUG
    checkCudaErrors(cuCtxSynchronize());
#endif

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    checkCudaErrors(cuMemcpyDtoH(h_C, deviceDataResult, size));

    // Verify result
    int i;

    for (i = 0; i < N; ++i) {
        float sum = h_A[i] + h_B[i];

        if (fabs(h_C[i] - sum) > 1e-7f) {
            break;
        }
    }

    checkCudaErrors(cuMemFree(deviceData1));
    checkCudaErrors(cuMemFree(deviceData2));
    checkCudaErrors(cuMemFree(deviceDataResult));

    // Free host memory
    if (h_A) {
        free(h_A);
    }

    if (h_B) {
        free(h_B);
    }

    if (h_C) {
        free(h_C);
    }

    printf("%s\n", (i == N) ? "Result = PASS" : "Result = FAIL");

    /*exit((i == N) ? EXIT_SUCCESS : EXIT_FAILURE);*/
}


END_BLUST_NAMESPACE
