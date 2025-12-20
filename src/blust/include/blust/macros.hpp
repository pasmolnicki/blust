#pragma once

#ifndef ENABLE_CUDA_BACKEND
#   define ENABLE_CUDA_BACKEND 0
#endif

#ifndef ENABLE_OPENCL_BACKEND
#   define ENABLE_OPENCL_BACKEND 0
#endif


#if ENABLE_CUDA_BACKEND && ENABLE_OPENCL_BACKEND
#   undef ENABLE_OPENCL_BACKEND
#endif


// Define the main backend name
#if ENABLE_CUDA_BACKEND
#   define BLUST_BACKEND_CUDA
#elif ENABLE_OPENCL_BACKEND
#   define BLUST_BACKEND_OPENCL
#else
#   define BLUST_BACKEND_CPU
#endif
