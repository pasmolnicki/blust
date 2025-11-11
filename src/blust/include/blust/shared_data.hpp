#pragma once

#include "types.hpp"


START_BLUST_NAMESPACE

template <typename dtype>
concept IsDType = std::is_floating_point_v<dtype> || std::is_same_v<dtype, CUdeviceptr>;

template <IsDType dtype>
class shared_data {
public:
    typedef CUdeviceptr cu_pointer;
    typedef CUdeviceptr& cu_pointer_ref;
    typedef dtype* pointer;
    typedef const dtype* const_pointer;

    typedef std::variant<cu_pointer, pointer> internal_data; 
    enum class pointer_type { buffer = 1, cuda = 2 };

    static int n_allocs;
    static int max_allocs;

    static void inc_alloc(int n = 1) {
        max_allocs = std::max(max_allocs, n_allocs += n); 
    }

    // Memory alignment
    static constexpr size_t alignment = 32;

    /**
     * @brief Get total size in bytes, with given alignment
     */
    template <size_t Alignment, typename dtype>
    static constexpr size_t get_bytesize(size_t count) noexcept
    {
        return ((count * sizeof(dtype) + Alignment - 1) / Alignment ) * Alignment;
    }

    /**
     * @brief Get total size in bytes
     */
    static constexpr size_t get_bytesize(size_t count) noexcept
    {
        return get_bytesize<alignment, dtype>(count);
    }

    /**
     * @brief Allocates memory with given alignment and number of elements
     */
    template <size_t Alignment>
    static inline pointer aligned_alloc(size_t count) noexcept
    {
        return static_cast<pointer>(std::aligned_alloc(Alignment, get_bytesize(count)));
    }

    /**
     * @brief Allocate aligned memory with given number of lemenets
     */
    static inline pointer aligned_alloc(size_t count) noexcept
    {
        return aligned_alloc<alignment>(count);
    }

    // Free the aligned memory
    static inline void aligned_free(pointer src) noexcept
    {
        std::free(src);
    }
};


template <IsDType dtype>
class heap_data {
public:
    typedef dtype* pointer;

    heap_data(shape dim, dtype init) {
        
    }

    void build(shape dim, dtype init) {

    }
};

template <IsDType dtype>
class cu_data {
public:
    typedef CUdeviceptr cu_pointer;

    cu_data(shape dim, dtype init) {
        // Create the data
    }

    cu_data(const cu_data& data) {
        // Make a full copy of the data
    }

    cu_data(cu_data&& data) {
        // Take ownership of the data
    }

private:
    cu_pointer ptr;
};


END_BLUST_NAMESPACE