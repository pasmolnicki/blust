cmake_minimum_required(VERSION 3.13)

project(BlustDL LANGUAGES C CXX)

# Options, used later in the build
# DEVICE_BACKEND_NAME: The device backend to use (cuda/opencl)
set(DEVICE_BACKEND_NAME "opencl" CACHE STRING "Device backend to use (cuda/opencl/none)")
# set(DEVICE_BACKEND_NAME "cuda" CACHE STRING "Device backend to use (cuda/opencl)")
message(STATUS "Using device backend: ${DEVICE_BACKEND_NAME}")
# Set C++ standard and compiler flags
set(CMAKE_CXX_STANDARD 20)
# GNU flags (g++/clang++)
set(GNU_RELEASE_FLAGS "-O3 -ffast-math -march=x86-64-v3 -fopenmp")
set(GNU_DEBUG_FLAGS "-O0 -g -march=x86-64-v3 -fsanitize=address -fno-omit-frame-pointer")
# MSVC flags
set(WIN_RELEASE_FLAGS "/O2 /arch:AVX2 /fp:fast")
set(WIN_DEBUG_FLAGS "/Od /Zi")

# Set the actual compiler flags based on the compiler
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${GNU_RELEASE_FLAGS}")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${GNU_DEBUG_FLAGS}")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${WIN_RELEASE_FLAGS}")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${WIN_DEBUG_FLAGS}")
endif()

# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
    message("Using default build type: Release")
    set(CMAKE_BUILD_TYPE Release)
else()
    message(STATUS "Using build type: ${CMAKE_BUILD_TYPE}")
endif()

# Set output directories, for some reason simply 
# setting CMAKE_RUNTIME_OUTPUT_DIRECTORY does not work
make_directory(${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Download gtest
include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

FetchContent_Declare(googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG main
)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing benchmark's tests" FORCE)   

# For Windows: Prevent overriding the parent project's compiler/linker settings
if(WIN32)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif()
FetchContent_MakeAvailable(googletest googlebenchmark)


# Resolve device backend dependencies
if(DEVICE_BACKEND_NAME STREQUAL "cuda" OR DEVICE_BACKEND_NAME STREQUAL "opencl")
    # Always prefer CUDA over OpenCL if both are available
    find_package(CUDAToolkit)
    find_package(OpenCL)
    
    if(DEVICE_BACKEND_NAME STREQUAL "cuda")
        if (CUDAToolkit_FOUND)
            set(DEVICE_BACKEND_NAME "cuda" CACHE STRING "Device backend to use (cuda/opencl)" FORCE)
            set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 72 75 80 86 87 89 90 100 101 120)
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
            set(CUDA_FATBIN_FILE "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/cuda_kernel64.fatbin")
            set(CUDA_KERNEL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/blust/src/cuda/cuda_kernel.cu")
            message(STATUS "Found CUDA Toolkit: ${CUDAToolkit_VERSION}")
        else()
            message(WARNING 
                "CUDA Toolkit not found. Falling back to OpenCL backend.")
            set(DEVICE_BACKEND_NAME "opencl" CACHE STRING "Device backend to use (cuda/opencl)" FORCE)
        endif()
    endif()
    
    if(DEVICE_BACKEND_NAME STREQUAL "opencl")
        if (OpenCL_FOUND)
            set(DEVICE_BACKEND_NAME "opencl" CACHE STRING "Device backend to use (cuda/opencl)" FORCE)
            message(STATUS "Found OpenCL: ${OpenCL_VERSION_MAJOR}.${OpenCL_VERSION_MINOR}")
        else()
            message(WARNING 
                "OpenCL not found. Falling back to host (CPU) backend.")
            set(DEVICE_BACKEND_NAME "none" CACHE STRING "Device backend to use (cuda/opencl)" FORCE)
        endif()
    endif()
endif()
    

# Load the library
add_subdirectory(src)

if (DEVICE_BACKEND_NAME STREQUAL "cuda")
    message(STATUS "Generating CUDA fatbin for device backend")
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17 cuda_std_17)
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    add_custom_command(
        OUTPUT ${CUDA_FATBIN_FILE}
        COMMAND ${CMAKE_CUDA_COMPILER} ${INCLUDES} ${ALL_CCFLAGS} -Wno-deprecated-gpu-targets  ${GENCODE_FLAGS} -o ${CUDA_FATBIN_FILE} -fatbin ${CUDA_KERNEL_SOURCE}
        DEPENDS ${CUDA_KERNEL_SOURCE}
        COMMENT "Building CUDA fatbin: ${CUDA_FATBIN_FILE}"
    )

    # Create a dummy target for fatbin generation
    add_custom_target(generate_fatbin ALL DEPENDS ${CUDA_FATBIN_FILE})

    # Ensure cuda driver depends on the fatbin
    add_dependencies(${PROJECT_NAME} generate_fatbin)

    # Define macro to enable CUDA backend
    # target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_CUDA_BACKEND=1)

elseif(DEVICE_BACKEND_NAME STREQUAL "opencl")
    # target_compile_definitions(${PROJECT_NAME} PRIVATE CL_TARGET_OPENCL_VERSION=100 ENABLE_OPENCL_BACKEND=1)
else()
    # Do nothing, using host (CPU) backend
endif()

install(TARGETS ${PROJECT_NAME})
# include(CTest)

# Tests
enable_testing()
add_subdirectory(tests)



# Benchmarking
# add_subdirectory(bench)
