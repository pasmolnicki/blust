#pragma once

#include "matrix/matrix.hpp"
#include "layers/Dense.hpp"
#include "layers/Input.hpp"
#include "models/Model.hpp"
#include "backend/cpu.hpp"
#include "backend/cuda_driver.hpp"
#include "backend/optimized.hpp"


START_BLUST_NAMESPACE

// Initialize the blust library, `device` can be either "cpu", "cuda" or "optimized"
inline void init(int argc, char** argv, std::string device = "")
{
	if (device == "cpu")
		g_backend = std::make_unique<cpu_backend>();
	else if (device == "cuda")
		g_backend = std::make_unique<cuda_backend>(argc, argv);
	else
		g_backend = std::make_unique<optimized_backend>(argc, argv);
}

END_BLUST_NAMESPACE