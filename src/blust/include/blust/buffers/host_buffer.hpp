#pragma once

#include "internal_tensor_data.hpp"

START_BLUST_NAMESPACE

template <IsDType dtype>
class tensor_host_buffer : public internal_tensor_data<dtype> {
public:
    struct AlignedDeleter {
        inline void operator()(dtype *p) { std::free(p); }
    };

    using unique_ptr = std::unique_ptr<dtype, AlignedDeleter>;
    using typename internal_tensor_data<dtype>::pointer;
    using typename internal_tensor_data<dtype>::const_pointer;
    using typename internal_tensor_data<dtype>::gen_fn;
    static constexpr auto alignment = internal_tensor_data<dtype>::alignment;

    // Construct new buffer
    tensor_host_buffer(size_t count, dtype init) {
        utils::inc_allocs(1);
        this->m_size = count;
        this->m_data.reset(utils::aligned_alloc<alignment, dtype>(count));
        this->m_bytesize = utils::get_bytesize<alignment, dtype>(count);
        std::fill_n(this->m_data.get(), count, init);
    }

    // Claim given pointer
    explicit
    tensor_host_buffer(pointer data, size_t count) {
        this->m_size = count;
        this->m_bytesize = utils::get_bytesize<alignment, dtype>(count);
        this->m_data.reset(data);
    }

    // Copy all data from given pointer
    explicit
    tensor_host_buffer(const_pointer data, size_t count) {
        utils::inc_allocs(1);
        this->m_size = count;
        this->m_bytesize = utils::get_bytesize<alignment, dtype>(count);
        this->m_data.reset(utils::aligned_alloc<alignment, dtype>(count));
        std::copy_n(data, this->m_size, m_data.get());
    }

    ~tensor_host_buffer() {
        // utils::inc_allocs(-1);
    }

    tensor_host_buffer* clone() {
        return new tensor_host_buffer(const_pointer(this->m_data.get()), this->m_size);
    }

    void fill(dtype init) noexcept {
        std::fill_n(m_data.get(), this->m_size, init);
    }

    void generate(gen_fn gen) noexcept {
        std::generate_n(m_data.get(), this->m_size, gen);
    }

    pointer begin() const noexcept {
        return m_data.get();
    }

    pointer end() const noexcept {
        return m_data.get() + this->m_size;
    }

    pointer data() noexcept {
        return m_data.get();
    }

    const_pointer data() const noexcept {
        return m_data.get();
    }

    pointer release() noexcept {
        this->m_bytesize = this->m_size = 0;
        return m_data.release();
    }
private:
    unique_ptr m_data{nullptr};
};

END_BLUST_NAMESPACE