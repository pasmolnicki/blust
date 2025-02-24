#include <blust/cuda/cuda_matrix.hpp>

START_BLUST_NAMESPACE

using namespace std;

CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction cu_vector_add;
CUfunction cu_vector_sub;
float* h_A;
float* h_B;
float* h_C;
CUdeviceptr d_A;
CUdeviceptr d_B;
CUdeviceptr d_C;

void run_test();
int CleanupNoFailure();
void RandomInit(float*, int);
bool findModulePath(const char*, string&, char**, string&);

// define input fatbin file
#ifndef FATBIN_FILE
#define FATBIN_FILE "cuda_kernel64.fatbin"
#endif

template <cuda_function_type type>
void lanuch_kernel(matrix_t& res, matrix_t& mat1, matrix_t& mat2)
{

}

// Initialize the cuda module
void cuda_init(int argc, char** argv) {
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
        exit(EXIT_FAILURE);
    }
    else {
        std::cout << "> initCUDA loading module: <" << module_path << ">\n";
    }

    // Empty
    if (!fatbin.str().size()) {
        std::cout << "fatbin file empty. exiting..\n";
        exit(EXIT_FAILURE);
    }

    // Create module from binary file (FATBIN)
    checkCudaErrors(cuModuleLoadData(&cuModule, fatbin.str().c_str()));

    // Load all functions
    checkCudaErrors(
        cuModuleGetFunction(&cu_vector_add, cuModule, "cu_vector_add"));
    checkCudaErrors(
        cuModuleGetFunction(&cu_vector_sub, cuModule, "cu_vector_sub"));

    run_test();
}



// Run test 
void run_test() {
    printf("TEST: Vector Addition (Driver API)\n");
    int N = 50000, devID = 0;
    size_t size = N * sizeof(float);

    // Allocate input vectors h_A and h_B in host memory
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    // Initialize input vectors
    RandomInit(h_A, N);
    RandomInit(h_B, N);

    // Allocate vectors in device memory
    checkCudaErrors(cuMemAlloc(&d_A, size));

    checkCudaErrors(cuMemAlloc(&d_B, size));

    checkCudaErrors(cuMemAlloc(&d_C, size));

    // Copy vectors from host memory to device memory
    checkCudaErrors(cuMemcpyHtoD(d_A, h_A, size));

    checkCudaErrors(cuMemcpyHtoD(d_B, h_B, size));

    
    // This is the new CUDA 4.0 API for Kernel Parameter Passing and Kernel
    // Launch (simpler method)

    // Grid/Block configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    void* args[] = { &d_A, &d_B, &d_C, &N };

    // Launch the CUDA kernel
    checkCudaErrors(cuLaunchKernel(cu_vector_add, blocksPerGrid, 1, 1,
        threadsPerBlock, 1, 1, 0, NULL, args, NULL));
    

#ifdef _DEBUG
    checkCudaErrors(cuCtxSynchronize());
#endif

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    checkCudaErrors(cuMemcpyDtoH(h_C, d_C, size));

    // Verify result
    int i;

    for (i = 0; i < N; ++i) {
        float sum = h_A[i] + h_B[i];

        if (fabs(h_C[i] - sum) > 1e-7f) {
            break;
        }
    }

    CleanupNoFailure();
    printf("%s\n", (i == N) ? "Result = PASS" : "Result = FAIL");

    exit((i == N) ? EXIT_SUCCESS : EXIT_FAILURE);
}

int CleanupNoFailure() {
    // Free device memory
    checkCudaErrors(cuMemFree(d_A));
    checkCudaErrors(cuMemFree(d_B));
    checkCudaErrors(cuMemFree(d_C));

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

    checkCudaErrors(cuCtxDestroy(cuContext));

    return EXIT_SUCCESS;
}
// Allocates an array with random float entries.
void RandomInit(float* data, int n) {
    for (int i = 0; i < n; ++i) {
        data[i] = rand() / (float)RAND_MAX;
    }
}


END_BLUST_NAMESPACE
