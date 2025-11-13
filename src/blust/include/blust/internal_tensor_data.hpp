#pragma once

#include <memory>
#include <type_traits>

#include "types.hpp"


START_BLUST_NAMESPACE

template <typename dtype>
concept IsDType = std::is_floating_point_v<dtype> || std::is_same_v<dtype, CUdeviceptr>;


template <IsDType dtype>
class internal_tensor_data {
public:
    typedef CUdeviceptr cu_pointer;
    typedef CUdeviceptr& cu_pointer_ref;
    typedef dtype* pointer;
    typedef const dtype* const_pointer;
    typedef std::function<dtype> gen_fn;

    static int n_allocs;
    static int max_allocs;

    static void inc_alloc(int n = 1) {
        max_allocs = std::max(max_allocs, n_allocs += n); 
    }

    // Memory alignment
    static constexpr size_t alignment = 32;

    // bytesize getter
    inline size_t get_bytesize() const noexcept {
        return m_bytesize;
    }

    // get total count (may not be equal to bytesize/sizeof(dtype))
    inline size_t size() const noexcept {
        return m_size;
    }

    // void build(size_t count, dtype init) = 0;
    virtual void fill(dtype v) = 0;
    virtual void generate(gen_fn gen) = 0;
protected:
    size_t m_bytesize{0};
    size_t m_size{0};
};

template <IsDType dtype>
class tensor_buffer : public internal_tensor_data<dtype> {
    std::unique_ptr<dtype> m_data{nullptr};
public:
    using typename internal_tensor_data<dtype>::pointer;
    using typename internal_tensor_data<dtype>::const_pointer;
    using typename internal_tensor_data<dtype>::gen_fn;

    heap_data(size_t count, dtype init) {
        m_size = count;
        m_data = utils::aligned_alloc<alignment, dtype>(count);
        m_bytesize = utils::get_bytesize<alignment, dtype>(count);
        std::fill_n(m_data.get(), count, init);
    }


    void fill(dtype init) {
        std::fill_n(m_data.get(), m_size, init);
    }

    void generate(gen_fn gen) {
        std::generate_n(m_data.get(), m_size, gen);
    }
};

template <IsDType dtype>
class tensor_cuda_buffer : internal_tensor_data<dtype> {
public:
    using typename internal_tensor_data<dtype>::cu_pointer;
    using typename internal_tensor_data<dtype>::gen_fn;

    cu_data(size_t count, dtype init) {
        // Create the data
    }

    void fill(dtype init) {
        // std::fill_n(m_data.get(), m_size, init);
    }

    void generate(gen_fn gen) {
        // std::generate_n(m_data.get(), m_size, gen);
    }

private:
    cu_pointer ptr{0};
};

END_BLUST_NAMESPACE