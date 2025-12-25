#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <algorithm>
#include <numeric>
#include <memory>
#include <variant>
#include <functional>

#include "utils.hpp"
#include "shape.hpp"
#include "buffers/data_handler.hpp"

START_BLUST_NAMESPACE

template <typename T>
concept IsPointerOrCU = std::is_same_v<T, number_t*> || std::is_same_v<T, CUdeviceptr>;

template <typename T>
concept TensorDataType = std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_integral_v<T>;


// Main tensor class, can either hold heap memory buffer, or gpu memory pointer
// The buffer is 32-byte aligned
class tensor 
{
public:
    friend class operations;
    friend class cpu_ops;
    friend class ops_tensor;

    using ntype = number_t;
    using cu_pointer = CUdeviceptr;
    using cu_pointer_ref = CUdeviceptr&;
    using pointer = number_t*;
    using const_pointer = const number_t*;
    using pointer_type = typename data_handler<number_t>::pointer_type;

    constexpr static auto alignment = internal_tensor_data<number_t>::alignment;

    // Default c'tor
    tensor() = default;

   /**
    * @brief Create a tensor object
    * @param dim dimensions of the tensor
    * @param init initial value for each 'cell' in a tensor
    */
    tensor(const shape& dim, number_t init = 0.0) noexcept : m_shape(dim) {
        m_handler.build(dim, init, pointer_type::host);
    }

    tensor(const shape& dim, number_t init, pointer_type type) noexcept : m_shape(dim) {
        m_handler.build(dim, init, type);
    }

    // Copy constructor
    tensor(const tensor& t) noexcept { void(*this = t); }

    // Move constructor
    tensor(tensor&& t) noexcept { void(*this = std::move(t)); }

    // Takes the ownership of the `data`, might blow your leg
    // tensor(pointer data, const shape& dim) noexcept : m_shape(dim) {
    //     m_handler.build(data, dim);
    // }

    // Copy dim.total() elements from the given data to the internal buffer
    tensor(const_pointer data, const shape& dim) noexcept : m_shape(dim) {
        m_handler.build(data, dim);
    }

    tensor& operator=(const tensor& t) noexcept {
        if (this == &t)
            return *this;

        m_handler = t.m_handler;
        m_shape = t.m_shape;
        return *this;
    }

    tensor& operator=(tensor&& t) noexcept {
        m_handler = std::move(t.m_handler);
        m_shape = std::move(t.m_shape);
        return *this;
    }

    // Allocate underlying tensor buffer filled with given `init` value
    void build(const shape& dim, number_t init = 0.0) noexcept {
        m_shape = dim;
        m_handler.build(dim, init, pointer_type::host);
    }

    void build(const shape& dim, number_t init, pointer_type type) noexcept {
        m_shape = dim;
        m_handler.build(dim, init, type);
    }

    // Get the dimensions (as a vector)
    shape::dim_t dim() const noexcept { return m_shape.dim(); }
    const shape& layout() const noexcept { return m_shape; }

    // Get the rank of the tensor
    size_t rank() const noexcept { return m_shape.rank(); }

    // Get total size of the internall buffer
    size_t size() const noexcept { return m_handler.size(); }

    // Get number of bytes memory holds (doesn't have to be sames as size*sizeof(number_t) since it's aligned)
    size_t bytesize() const noexcept { return m_handler.bytesize(); }

    // Get buffer type
    pointer_type type() const noexcept { return m_handler.type(); }

    auto& handler() noexcept { return m_handler; }
    const auto& handler() const noexcept { return m_handler; }

    void to_host() noexcept { m_handler.to_host(); }

    // Check wheter internal buffer is stored in gpu memory
    bool is_cuda() const noexcept { return m_handler.is_cuda(); }

    // fill the tensor with given predicate
    void fill(std::function<number_t()> f) noexcept {
        m_handler.generate(f);
    }

    // fill the tensor with given value
    void fill(number_t val) noexcept {
        m_handler.fill(val);
    }

    // Check if the tensor is empty
    bool empty() const noexcept {
        return m_handler.empty();
    }

    // Get the internal 1d buffer
    pointer data() noexcept { return m_handler.data();}
    const_pointer data() const noexcept { return m_handler.data();}

    pointer begin() noexcept { return m_handler.begin(); }
    auto begin() const noexcept { return m_handler.begin(); }
    pointer end() noexcept { return m_handler.end(); }
    auto end() const noexcept { return m_handler.end(); }

    // Release the buffer, should be wrapped in a unique pointer with array type
    pointer release() noexcept { return m_handler.release(); }
    
    // Print the tensor to output stream
    friend std::ostream& operator<<(std::ostream& out, const tensor& t) noexcept;

    // Array like access
    number_t& operator()(size_t i) { return *(data() + i); }
    number_t operator()(size_t i) const { return *(data() + i); }

private:

    // Make shared tensor buffer
    tensor(data_handler<number_t>& handler, shape& dim):
        m_handler(handler.make_shared()), m_shape(dim) {}

    // Private constructor for optimized cuda buffer management
    tensor(cu_pointer cu_ptr, shape dim) noexcept : m_shape(dim) {
        m_handler.build(cu_ptr, dim);
    }

    cu_pointer cu_release() noexcept { return m_handler.cu_release(); }
    cu_pointer cu_data() const noexcept { return m_handler.cu_data(); }
    // cu_pointer_ref cu_data() noexcept { return m_handler.cu_data(); }

    shape m_shape{};
    data_handler<number_t> m_handler;
    
    /**
     * @brief Borrow the buffer from given tensor, will share the buffer with 't'
     */
    inline void M_borrow(const tensor& t) noexcept
    {
        m_shape   = t.m_shape;
        m_handler = t.m_handler.make_shared();
    }

    // Print the tensor recursively, to given output stream, rank = t.rank(), index = 0, offset = 0
    static void M_print_tensor(
        const tensor& t, std::ostream& out, size_t rank, 
        size_t index = 0, size_t offset = 0
    ) noexcept;
};

END_BLUST_NAMESPACE

#endif //TENSOR_HPP
