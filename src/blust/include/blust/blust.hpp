#pragma once

#include "matrix/matrix.hpp"
#include "layers/Dense.hpp"
#include "layers/Input.hpp"
#include "models/Model.hpp"
#include "models/Sequential.hpp"
#include "backend/cpu.hpp"
#include "backend/cuda_driver.hpp"
#include "backend/optimized.hpp"
#include "settings.hpp"
#include "datasets/mnist.hpp"
#include "backend/operations.hpp"
#include "tensor.hpp"
#include "backend/ops.hpp"

START_BLUST_NAMESPACE

// Initialize the blust library, `device` can be either "cpu", "cuda" or "optimized"
inline void init(int argc, char** argv, std::string device = "")
{
	g_settings = std::make_unique<settings>();
	g_settings->init(argv);
	ops = std::make_unique<cpu_ops>(8);

// 	if (device == "cpu")
// 		g_backend = std::make_unique<cpu_backend>();
// 	else if (device == "cuda")
// 		g_backend = std::make_unique<cuda_backend>(argc, argv);
// 	else 
// 		g_backend = std::make_unique<optimized_backend>(argc, argv);
}

END_BLUST_NAMESPACE