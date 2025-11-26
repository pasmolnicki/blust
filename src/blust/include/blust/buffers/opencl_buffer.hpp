#pragma once
#include "internal_tensor_data.hpp"

START_BLUST_NAMESPACE

template <IsDType dtype>
class tensor_opencl_buffer : public internal_tensor_data<dtype> {
public:
    using host_pointer = dtype*;
    using const_host_pointer = const dtype*;
    using typename internal_tensor_data<dtype>::gen_fn;

#if !ENABLE_OPENCL_BACKEND

    using pointer = dtype*;
    using const_pointer = const dtype*;

    tensor_opencl_buffer(size_t count, dtype init) {}

    tensor_opencl_buffer* clone() { return nullptr; }
    void fill(dtype init) noexcept {}
    void generate(gen_fn gen) noexcept {}
    pointer data() noexcept { return nullptr; }
    pointer release() noexcept { return nullptr; }
#else

    using pointer = cl::Buffer;
    using const_pointer = const cl::Buffer;

    // Create new buffer
    tensor_opencl_buffer(size_t count, dtype init) : 
    m_context(g_settings->opencl_context().context()) {
        std::vector<dtype> temp(count, init);
        // Create the data
        m_ptr = cl::Buffer(
            m_context,
            temp.begin(),
            temp.end(),
            true
        );
        this->m_bytesize = sizeof(dtype) * count;
        this->m_size = count;
    }

    // Copy host buffer
    tensor_opencl_buffer(const_host_pointer data, size_t count) :
    m_context(g_settings->opencl_context().context()) {
        m_ptr = cl::Buffer(m_context, data, data + count, true);
        this->m_bytesize = sizeof(dtype) * count;
        this->m_size = count;        
    }

    // Copy the device buffer
    explicit
    tensor_opencl_buffer(const cl::Buffer& buffer, size_t count) :
    m_context(g_settings->opencl_context().context()) {
        // Deep copy the buffer
        
        this->m_bytesize = sizeof(dtype) * count;
        this->m_size = count;
    }

    // Claim pointer
    explicit
    tensor_opencl_buffer(cl::Buffer&& buffer, size_t count) :
    m_context(g_settings->opencl_context().context()) {
        m_ptr = buffer;
        this->m_bytesize = sizeof(dtype) * count;
        this->m_size = count;
    }

    ~tensor_opencl_buffer() {
        // Clean up the buffer
    }

    tensor_opencl_buffer* clone() {
        return new tensor_opencl_buffer(m_ptr, this->m_size);
    }

    void fill(dtype init) noexcept {
        // std::fill_n(m_data.get(), m_size, init);
        // Could create a custom iterator, that fills on the fly
        std::vector<dtype> temp(this->m_size, init);
        m_ptr = cl::Buffer(
            m_context, 
            temp.begin(),
            temp.end(),
            true
        );
    }

    void generate(gen_fn gen) noexcept {
        // This is quite expensive, for now simply create a temporary buffer,
        // fill it, and then copy to device
        std::vector<dtype> temp(this->m_size);
        std::generate_n(temp.begin(), this->m_size, gen);
        // Copy to device
        m_ptr = cl::Buffer(m_context, temp.begin(), temp.end(), true);
    }

    cl::Buffer& data() noexcept {
        return m_ptr;
    }

    cl::Buffer release() noexcept {
        auto ret = m_ptr;
        m_ptr = cl::Buffer();
        this->m_size = this->m_bytesize = 0;
        return ret;
    }
private:
    cl::Context m_context; // the device to copy to
    cl::Buffer m_ptr; // actual buffer
#endif
};

END_BLUST_NAMESPACE