#pragma once

#include "internal_tensor_data.hpp"

START_BLUST_NAMESPACE

template <IsDType dtype>
class tensor_cuda_buffer : public internal_tensor_data<dtype> {
public:
    using typename internal_tensor_data<dtype>::cu_pointer;
    using typename internal_tensor_data<dtype>::gen_fn;

    // Claim pointer
    explicit
    tensor_cuda_buffer(cu_pointer data, size_t count) {
        ptr = data;
        this->m_bytesize = this->m_size = count;
    }

    tensor_cuda_buffer(size_t count, dtype init) {
        // Create the data
    }

    ~tensor_cuda_buffer() {
        // Clean up the buffer
    }

    tensor_cuda_buffer* clone() {
        return new tensor_cuda_buffer(size_t(0), 0);
    }

    void fill(dtype init) noexcept {
        // std::fill_n(m_data.get(), m_size, init);
    }

    void generate(gen_fn gen) noexcept {
        // std::generate_n(m_data.get(), m_size, gen);
    }

    cu_pointer data() const noexcept {
        return ptr;
    }

    cu_pointer release() noexcept {
        auto ret = ptr;
        ptr = 0;
        this->m_bytesize = this->m_size = 0;
        return ret;
    }

    void memcpy(internal_tensor_data<dtype>* other) override {
        tensor_cuda_buffer<dtype>* other_cuda = dynamic_cast<tensor_cuda_buffer<dtype>*>(other);
        if (this->m_size != other->size() || other_cuda == nullptr) {
            return;
        }
        
        // Copy device to device
    }

private:
    cu_pointer ptr{0};
};

END_BLUST_NAMESPACE