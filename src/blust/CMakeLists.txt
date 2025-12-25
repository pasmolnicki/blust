add_library(blust)

# Add defines - USE_CUDA_BACKEND is defined in the main CMakeLists file as option

add_subdirectory(datasets)

if (DEVICE_BACKEND_NAME STREQUAL "cuda")
	target_compile_definitions(blust PRIVATE ENABLE_CUDA_BACKEND=1)
elseif (DEVICE_BACKEND_NAME STREQUAL "opencl")
	target_compile_definitions(blust PRIVATE CL_TARGET_OPENCL_VERSION=200 ENABLE_OPENCL_BACKEND=1)
endif()


target_sources(blust PRIVATE
	"src/models/Model.cpp"
	"src/cuda/cuda_driver.cpp"
	"src/datasets/mnist.cpp"
	"src/operations/cpu.cpp"
	"src/operations/nn_ops.cpp"
	"src/operations/opencl.cpp"
	"src/operations/operations.cpp"
	"src/tensor.cpp"
	"src/settings.cpp"
)

if(DEVICE_BACKEND_NAME STREQUAL "cuda")
	message(STATUS "Linking cuda driver library")
	target_link_libraries(blust PUBLIC CUDA::cuda_driver)
elseif(DEVICE_BACKEND_NAME STREQUAL "opencl")
	message(STATUS "Copying OpenCL kernel file and linking OpenCL library")
	target_link_libraries(blust PUBLIC OpenCL)
	configure_file(${CMAKE_CURRENT_LIST_DIR}/src/opencl/opencl-kernel.cl ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} COPYONLY)
endif()

target_include_directories(blust PUBLIC include)