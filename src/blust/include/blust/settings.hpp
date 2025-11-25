#pragma once

#include <memory>
#include <filesystem>

#include "namespaces.hpp"
#include "buffers/opencl_buffer_context.hpp"

START_BLUST_NAMESPACE

class settings
{
#if ENABLE_OPENCL_BACKEND
	opencl_buffer_context m_buffer_context;
#endif

	std::filesystem::path m_path;
public:
	constexpr auto version() const noexcept { return "0.0.1"; }
	static constexpr const char* backend() noexcept {
#if defined(BLUST_BACKEND_CUDA)
		return "CUDA";
#elif defined(BLUST_BACKEND_OPENCL)
		return "OpenCL";
#else
		return "CPU";
#endif
	}

	settings() = default;

	void init(char** argv)
	{
		m_path = std::filesystem::path(argv[0]).parent_path();
	}

	std::filesystem::path& path() noexcept { return m_path; }


#if ENABLE_OPENCL_BACKEND
	opencl_buffer_context& opencl_context() noexcept { return m_buffer_context; }
#endif
};

extern std::unique_ptr<settings> g_settings;

END_BLUST_NAMESPACE