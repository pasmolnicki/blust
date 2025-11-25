#pragma once

#include "layers/Dense.hpp"
#include "layers/Input.hpp"
#include "models/Model.hpp"
#include "models/Sequential.hpp"
#include "backend/cuda_driver.hpp"
#include "settings.hpp"
#include "datasets/mnist.hpp"
#include "backend/cpu_ops.hpp"
#include "backend/opencl_ops.hpp"
#include "tensor.hpp"

START_BLUST_NAMESPACE

// Initialize the blust library, `device` can be either "cpu", "cuda" or "opencl"
inline void init(int argc, char** argv, std::string device = "")
{
	g_settings = std::make_unique<settings>();
	g_settings->init(argv);
	ops = std::make_unique<cpu_ops>(8);
}

END_BLUST_NAMESPACE