#pragma once

#include <blust/base_types.hpp>

class opencl_buffer_context {
public:
    opencl_buffer_context() = default;

#if ENABLE_OPENCL_BACKEND
    opencl_buffer_context(const cl::Context& context, const cl::CommandQueue& queue)
        : m_context_ptr(std::make_unique<cl::Context>(context)),
          m_queue_ptr(std::make_unique<cl::CommandQueue>(queue)) {}

    opencl_buffer_context(opencl_buffer_context&& other) noexcept
        : m_context_ptr(std::move(other.m_context_ptr)),
          m_queue_ptr(std::move(other.m_queue_ptr)) {}

    opencl_buffer_context& operator=(opencl_buffer_context&& other) noexcept {
        if (this != &other) {
            m_context_ptr = std::move(other.m_context_ptr);
            m_queue_ptr = std::move(other.m_queue_ptr);
        }
        return *this;
    }

    cl::Context& context() noexcept { return *m_context_ptr; }
    cl::CommandQueue& queue() noexcept { return *m_queue_ptr; }
private:
    std::unique_ptr<cl::Context> m_context_ptr;
    std::unique_ptr<cl::CommandQueue> m_queue_ptr;
#endif
};