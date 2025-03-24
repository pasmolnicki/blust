#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <algorithm>
#include <numeric>
#include <memory>

#include <cuda.h>

#include "utils.hpp"
#include "shape.hpp"

START_BLUST_NAMESPACE


// Main tensor class, can either hold heap memory buffer, or gpu memory pointer
// The buffer is 16-byte aligned
class tensor 
{
public:
    friend class operations;
    friend class cpu_ops;
    friend class ops_tensor;

    typedef CUdeviceptr cu_pointer;
    typedef number_t* pointer;
    typedef const pointer const_pointer;
    typedef union { CUdeviceptr cu_ptr; pointer data; } internal_data;
    enum class data_type { buffer = 1, cuda = 2 };

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
    template <size_t Alignment>
    static constexpr size_t get_bytesize(size_t count) noexcept
    {
        return ((count * sizeof(number_t) + Alignment - 1) / Alignment ) * Alignment;
    }

    /**
     * @brief Get total size in bytes
     */
    static constexpr size_t get_bytesize(size_t count) noexcept
    {
        return get_bytesize<alignment>(count);
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

    // Default c'tor
    tensor() = default;

   /**
    * @brief Create a tensor object
    * @param dim dimensions of the tensor
    * @param init initial value for each 'cell' in a tensor
    */
    tensor(const shape& dim, number_t init = 0.0) noexcept
    : m_shape(dim)
    {
        inc_alloc(1);
        auto count      = dim.total();
        m_bytesize      = get_bytesize(count);
        m_tensor.data   = aligned_alloc(count);

        if (init != 0.0)
            std::fill_n(m_tensor.data, count, init);
    }

    // Copy constructor
    tensor(const tensor& t) noexcept { void(*this = t); }

    // Move constructor
    tensor(tensor&& t) noexcept { void(*this = std::forward<tensor>(t)); }

    // Takes the ownership of the `data`, might blow your leg
    tensor(pointer data, const shape& dim) noexcept : m_shape(dim)
    {
        m_tensor.data   = data;
        m_data_type     = data_type::buffer;
        m_bytesize      = get_bytesize(m_shape.total());
    }

    tensor& operator=(const tensor& t) noexcept;
    tensor& operator=(tensor&& t) noexcept;

    virtual ~tensor() noexcept { M_cleanup_buffer(); }

    // Get the dimensions (as a vector)
    shape::dim_t dim() const noexcept { return m_shape.dim(); }
    const shape& layout() const noexcept { return m_shape; }

    // Get the rank of the tensor
    size_t rank() const noexcept { return m_shape.rank(); }

    // Get total size of the internall buffer
    size_t size() const noexcept { return m_shape.total(); }

    // Get number of bytes memory holds (doesn't have to be sames as size*sizeof(number_t) since it's aligned)
    size_t bytesize() const noexcept { return m_bytesize; }

    // Get buffer type
    data_type type() const noexcept { return m_data_type; }

    // Check wheter internal buffer is stored in gpu memory
    bool is_cuda() const noexcept { return m_data_type == data_type::cuda; }

    // Check if the tensor is empty
    bool empty() const noexcept 
    {
        return m_shape.m_dims.empty() || (
            m_data_type == data_type::buffer ? m_tensor.data == nullptr : m_tensor.cu_ptr == 0
        ); 
    }

    // Get the internal 1d buffer
    pointer data() noexcept { return m_tensor.data; }
    const_pointer data() const noexcept { return m_tensor.data; }

    // Release the buffer, should be wrapped in a unique pointer with array type
    pointer release() noexcept { return M_release_t<pointer>(); }
    
    // Print the tensor to output stream
    inline friend std::ostream& operator<<(std::ostream& out, const tensor& t) noexcept;

private:

    // Private constructor for optimized cuda buffer management
    tensor(cu_pointer cu_ptr, shape dim) noexcept : m_shape(dim) 
    {
        m_tensor.cu_ptr = cu_ptr;
        m_data_type     = data_type::cuda;
        m_bytesize      = get_bytesize(m_shape.total());
    }

    cu_pointer cu_release() noexcept { return M_release_t<cu_pointer>(); }
    cu_pointer cu_data() const noexcept { return m_tensor.cu_ptr; }

    shape m_shape{};
    internal_data m_tensor{0};
    data_type m_data_type{data_type::buffer};
    size_t m_bytesize{};
    bool m_shared{false};


    inline size_t M_alloc_buffer(const tensor& t) noexcept;
    
    /**
     * @brief Borrow the buffer from given tensor, will share the buffer with 't'
     */
    inline void M_borrow(const tensor& t) noexcept
    {
        m_shape     = t.m_shape;
        m_tensor    = t.m_tensor;
        m_data_type = t.m_data_type;
        m_bytesize  = t.m_bytesize;
        m_shared    = true;
    }

    // Delete the buffer if not shared
    inline void M_cleanup_buffer() noexcept
    {
        if (m_shared)
            return;
    
        if (m_data_type == data_type::buffer && m_tensor.data != nullptr) {
            aligned_free(m_tensor.data);
            m_tensor.data = nullptr;
            inc_alloc(-1);
        }
    }

    // Print the tensor recursively, to given output stream, rank = t.rank(), index = 0, offset = 0
    static void M_print_tensor(
        const tensor& t, std::ostream& out, size_t rank, 
        size_t index = 0, size_t offset = 0
    ) noexcept;


    // Get the internal buffer, either as a `pointer` or `cu_pointer`
    template <typename T>
    inline std::enable_if_t<std::is_same_v<T, pointer> || std::is_same_v<T, cu_pointer>, T>
    M_release_t() noexcept
    {
        T res;
        if constexpr (std::is_same_v<T, pointer>) { 
            res = m_tensor.data; 
            m_tensor.data = nullptr; 
        }
        else { 
            res = m_tensor.cu_ptr; 
            m_tensor.cu_ptr = 0; 
        }
        m_bytesize = 0;
        m_shape.clear();
        return res;
    }
};

END_BLUST_NAMESPACE

#endif //TENSOR_HPP
