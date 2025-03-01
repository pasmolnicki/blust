#pragma once

#include "namespaces.hpp"

#include <filesystem>

START_BLUST_NAMESPACE

class settings
{
	std::filesystem::path m_path;
public:
	constexpr auto version() const noexcept { return "0.0.1"; }

	settings() = default;

	void init(char** argv)
	{
		m_path = std::filesystem::path(argv[0]).parent_path();
	}

	std::filesystem::path& path() noexcept { return m_path; }
};

static std::unique_ptr<settings> g_settings;

END_BLUST_NAMESPACE