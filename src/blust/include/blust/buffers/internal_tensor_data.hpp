#pragma once

#include <functional>
#include <memory>
#include <type_traits>

#include <blust/settings.hpp>
#include <blust/utils.hpp>
#include <blust/base_types.hpp>

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
    typedef std::function<dtype()> gen_fn;

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
    virtual void fill(dtype v) noexcept = 0;
    virtual void generate(gen_fn gen) noexcept = 0;
    virtual void memcpy(internal_tensor_data<dtype>* other) = 0;
protected:
    size_t m_bytesize{0};
    size_t m_size{0};
};

END_BLUST_NAMESPACE