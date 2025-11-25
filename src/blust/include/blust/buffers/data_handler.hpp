#pragma once

#include <variant>
#include <memory>
#include <type_traits>

#include <blust/base_types.hpp>

// buffer implementations
#include "host_buffer.hpp"
#include "cuda_buffer.hpp"
#include "opencl_buffer.hpp"
#include "internal_tensor_data.hpp"

START_BLUST_NAMESPACE

template <IsDType dtype>
class data_handler {
public:
    typedef enum class pointer_type { host = 1, cuda = 2, opencl = 4 } pointer_type;
    typedef CUdeviceptr cu_pointer;
    typedef CUdeviceptr& cu_pointer_ref;
    using cl_pointer = tensor_opencl_buffer<dtype>::pointer;
    typedef dtype* pointer;
    typedef const dtype* const_pointer;
    using base_ptr = internal_tensor_data<dtype>*;
    using shared_host_ptr = std::shared_ptr<tensor_host_buffer<dtype>>;
    using shared_cu_ptr = std::shared_ptr<tensor_cuda_buffer<dtype>>;
    using shared_cl_ptr = std::shared_ptr<tensor_opencl_buffer<dtype>>;
    using variant_data = std::variant<shared_host_ptr, shared_cu_ptr, shared_cl_ptr>;

    data_handler() = default;

    data_handler(shape dim, dtype init, pointer_type type) {
        build(std::forward<shape>(dim), init, type);
    }

    data_handler(const data_handler<dtype>& other) {
        void(*this = other);
    }

    data_handler(data_handler<dtype>&& other) {
        void(*this = std::forward<data_handler<dtype>>(other));
    }

    ~data_handler() {
        void(0);
    }

    // Copies all conent of the `other` allocated memory
    data_handler<dtype>& operator=(const data_handler<dtype>& other) {
        // m_data = other.m_data;
        m_type = other.m_type;
        if (m_type == pointer_type::host) {
            m_data = shared_host_ptr(
                std::get<shared_host_ptr>(other.m_data)->clone());
        } 
        else if (m_type == pointer_type::opencl) {
            m_data = shared_cl_ptr(
                std::get<shared_cl_ptr>(other.m_data)->clone());
        }
        else {
            m_data = shared_cu_ptr(
                std::get<shared_cu_ptr>(other.m_data)->clone());
        }
        M_set_base_ptr();
        return *this;
    }

    // Moves all other's conent into this object
    inline data_handler<dtype>& operator=(data_handler<dtype>&& other) {
        m_data = std::move(other.m_data);
        m_type = other.m_type;
        M_set_base_ptr();
        return *this;
    }

    /**
     * Take ownership of the given buffer pointer
     */
    inline void build(pointer data, shape dim) noexcept {
        m_type = pointer_type::host;
        m_data = shared_host_ptr(new tensor_host_buffer<dtype>(data, dim.total()));
        M_set_base_ptr();
    }

    /**
     * Deep copy the `data` contents
     */
    inline void build(const_pointer data, shape dim) noexcept {
        m_type = pointer_type::host;
        m_data = shared_host_ptr(new tensor_host_buffer<dtype>(data, dim.total()));
        M_set_base_ptr();
    }

    inline void build(cu_pointer data, shape dim) noexcept {
        m_type = pointer_type::cuda;
        m_data = shared_cu_ptr(new tensor_cuda_buffer<dtype>(data, dim.total()));
        M_set_base_ptr();
    }

    /**
     * @brief Allocate internal buffer given dims and initial value
     */
    inline void build(shape dim, dtype init, pointer_type type) noexcept {
        m_type = type;
        if (type == pointer_type::cuda) {
            m_data = shared_cu_ptr(
                new tensor_cuda_buffer<dtype>(std::move(dim.total()), init));
        } 
        else if (type == pointer_type::opencl) {
            m_data = shared_cl_ptr(
                new tensor_opencl_buffer<dtype>(std::move(dim.total()), init));
        }
        else {
            m_data = shared_host_ptr(
                new tensor_host_buffer<dtype>(std::move(dim.total()), init));
        }
        M_set_base_ptr();
    }

    // Create new `data_handler` with shared internal data
    inline data_handler make_shared() const noexcept {
        return data_handler(m_data, m_type, m_base_ptr);
    }

    /**
     * @brief Make sure the underlying buffer is not shared 
     * (the atomic count of shared ptr == 1)
     */
    inline void ensure_unique() noexcept {
        if (m_type == pointer_type::host) {
            auto& sp = std::get<shared_host_ptr>(m_data);

            if (!sp || sp.use_count() == 1) {
                return;
            }

            m_data = shared_host_ptr(sp->clone());
            // auto clone = std::make_shared<tensor_host_buffer<dtype>>(sp->size(), dtype{});
            // std::copy_n(sp->data(), sp->size(), clone->data());
            // sp = std::move(clone);
        } 
        else if (m_type == pointer_type::opencl) {
            auto& sp = std::get<shared_cl_ptr>(m_data);
            
            if (!sp || sp.use_count() == 1) {
                return;
            }
            
            m_data = shared_cl_ptr(sp->clone());
            // clone OpenCL buffer (device-to-device copy)
            // auto clone = std::make_shared<tensor_opencl_buffer<dtype>>(sp->size(), dtype{});
            // cl::CommandQueue queue = ...; // Obtain the command queue
            // queue.enqueueCopyBuffer(sp->data(), clone->data(), 0, 0, bytes);
            // sp = std::move(clone);
        }
        else if (m_type == pointer_type::cuda) {
            auto& sp = std::get<shared_cu_ptr>(m_data);
            
            if (!sp || sp.use_count() == 1) {
                return;
            }
            
            m_data = shared_cu_ptr(sp->clone());
            // clone CUDA buffer (cuda-to-cuda copy)
            // auto clone = std::make_shared<tensor_cuda_buffer<dtype>>(sp->size(), dtype{});
            // cudaMemcpy(clone->data(), sp->data(), bytes, cudaMemcpyDeviceToDevice);
            // sp = std::move(clone);
        }
        M_set_base_ptr();
    }

    /**
     * @brief Fill internal buffer with given value
     */
    inline void fill(dtype init) noexcept {
        m_base_ptr->fill(init);
    }

    /**
     * @brief Apply given generator to each element of the tensor
     */
    inline void generate(std::function<dtype()> gen) noexcept {
        m_base_ptr->generate(gen);
    }

    inline bool empty() const noexcept {
        return m_base_ptr == nullptr || m_base_ptr->size() == 0;
    }

    inline size_t size() const noexcept {
        return m_base_ptr->size();
    }

    inline size_t bytesize() const noexcept {
        return m_base_ptr->get_bytesize();
    }

    inline bool is_host() const noexcept {
        return m_type == pointer_type::host;
    }

    inline bool is_opencl() const noexcept {
        return m_type == pointer_type::opencl;
    }

    inline bool is_cuda() const noexcept {
        return m_type == pointer_type::cuda;
    }

    void to_host() noexcept {
        if (is_host()) {
            return;
        }

        // Create new host buffer and copy data
        auto new_host_buffer = shared_host_ptr(
            new tensor_host_buffer<dtype>(size(), 0));
        
        if (is_cuda()) {
#if ENABLE_CUDA_BACKEND
            // Copy from CUDA to host
            cudaMemcpy(
                new_host_buffer->data(),
                cu_data(),
                bytesize(),
                cudaMemcpyDeviceToHost
            );
#endif
        }

        else if (is_opencl()) {
#if ENABLE_OPENCL_BACKEND
            // Copy from OpenCL to host
            cl::CommandQueue queue = g_settings->opencl_context().queue();
            queue.enqueueReadBuffer(
                cl_data(),
                CL_TRUE,
                0,
                bytesize(),
                new_host_buffer->data()
            );
#endif
        }

        // Update internal data to host buffer
        m_data = new_host_buffer;
        m_type = pointer_type::host;
        M_set_base_ptr();
    }

    pointer_type type() const noexcept {
        return m_type;
    }

    cu_pointer cu_data() const noexcept {
        return std::get<shared_cu_ptr>(m_data)->data();
    }

    const const_pointer data() const noexcept {
        return std::get<shared_host_ptr>(m_data)->data();
    }

    pointer data() noexcept {
        if (is_cuda() || is_opencl()) {
            to_host();
        }

        return std::get<shared_host_ptr>(m_data)->data();
    }

    pointer begin() const noexcept {
        return std::get<shared_host_ptr>(m_data)->begin();
    }

    pointer end() const noexcept {
        return std::get<shared_host_ptr>(m_data)->end();
    }

    pointer release() noexcept {
        if (is_cuda() || is_opencl()) {
            to_host();
        }

        return std::get<shared_host_ptr>(m_data)->release();
    }

    cu_pointer cu_release() noexcept {
        return std::get<shared_cu_ptr>(m_data)->release();
    }

    cl_pointer& cl_data() noexcept {
        return std::get<shared_cl_ptr>(m_data)->data();
    }

    cl_pointer cl_release() noexcept {
        return std::get<shared_cl_ptr>(m_data)->release();
    }

    template <typename T>
    T& get_internal_buffer() noexcept {
        if constexpr (std::is_same_v<T, tensor_host_buffer<dtype>>) {
            return *std::get<shared_host_ptr>(m_data);;
        }
        else if constexpr (std::is_same_v<T, tensor_cuda_buffer<dtype>>) {
            return *std::get<shared_cu_ptr>(m_data);;
        }
        else if constexpr (std::is_same_v<T, tensor_opencl_buffer<dtype>>) {
            return *std::get<shared_cl_ptr>(m_data);;
        }
        else {
            static_assert("Unsupported buffer type");
        }
    }

private:

    // Create shared data_handler
    data_handler(const variant_data& data, pointer_type type, base_ptr p) :
        m_data(data), m_type(type), m_base_ptr(p) {
            utils::inc_shared(1);
        }

    void M_set_base_ptr() noexcept {
        if (std::holds_alternative<shared_host_ptr>(this->m_data)) {
            this->m_base_ptr = std::get<shared_host_ptr>(this->m_data).get();
        } else if (std::holds_alternative<shared_cl_ptr>(this->m_data)) {
            this->m_base_ptr = std::get<shared_cl_ptr>(this->m_data).get();
        } else {
            this->m_base_ptr = std::get<shared_cu_ptr>(this->m_data).get();
        }
    }

    variant_data m_data{shared_host_ptr{nullptr}};
    pointer_type m_type{pointer_type::host};
    base_ptr m_base_ptr{nullptr};
};

END_BLUST_NAMESPACE