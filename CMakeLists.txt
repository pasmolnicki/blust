cmake_minimum_required(VERSION 3.13)

project(BlustDL LANGUAGES C CXX CUDA)

option(USE_CUDA_BACKEND "Whether to use a cuda device as computational backend, if disabled uses cpu" ON)
message("Cuda: ${USE_CUDA_BACKEND}")


set(CMAKE_CXX_STANDARD 20)
set(GNU_RELEASE_FLAGS "-O3 -ffast-math -march=native -fopenmp")
# set(GNU_RELEASE_FLAGS "-O0")
set(WIN_RELEASE_FLAGS "/O2 /arch:AVX2 /fp:fast")
set(GNU_DEBUG_FLAGS "-O0 -g -march=x86-64-v3 -fsanitize=address -fno-omit-frame-pointer")
set(WIN_DEBUG_FLAGS "/Od /Zi")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${GNU_RELEASE_FLAGS}")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${GNU_DEBUG_FLAGS}")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${WIN_RELEASE_FLAGS}")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${WIN_DEBUG_FLAGS}")
endif()


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


# cuda

if(USE_CUDA_BACKEND)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 72 75 80 86 87 89 90 100 101 120)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
    set(CUDA_FATBIN_FILE "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/cuda_kernel64.fatbin")
    set(CUDA_KERNEL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/blust/src/cuda/cuda_kernel.cu")
endif()

# Load the library
add_subdirectory(src)


if (USE_CUDA_BACKEND)
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
endif()

install(TARGETS ${PROJECT_NAME})
include(CTest)

# Tests
enable_testing()
add_subdirectory(tests)



# Benchmarking
add_subdirectory(bench)
