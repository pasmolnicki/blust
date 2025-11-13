#pragma once

#include <variant>
#include <memory>
#include <type_traits>

#include "types.hpp"
#include "internal_tensor_data.hpp"

START_BLUST_NAMESPACE

template <IsDType dtype>
class data_handler {
public:
    typedef enum {cuda_data, buffer_data} internal_type;
    typedef CUdeviceptr cu_pointer;
    typedef CUdeviceptr& cu_pointer_ref;
    typedef dtype* pointer;
    typedef const dtype* const_pointer;
    typedef std::shared_ptr<tensor_buffer<dtype>> shared_buffer_ptr;
    typedef std::shared_ptr<tensor_cuda_buffer<dtype>> shared_cu_ptr;
    typedef std::variant<shared_buffer_ptr, shared_cu_ptr> variant_data;

    data_handler() = default;

    data_handler(shape dim, dtype init, internal_type type) {
        build(std::move(dim), init, type);
    }

    data_handler(const data_handler<dtype>& other) {
        m_type = other.m_type;
        m_data = other.m_data;
    }

    data_handler(data_handler&& other) {
        m_type = other.m_type;
        m_data = std::move(other.m_data);
    }

    /**
     * @brief Allocate internal buffer given dims and initial value
     */
    inline void build(shape dim, dtype init, internal_type type) {
        m_type = type;
        if (type == cuda_data) {
            m_data = shared_cu_ptr(
                new tensor_cuda_buffer<dtype>(std::move(dim.total()), init));
        } else {
            m_data = shared_buffer_ptr(
                new tensor_buffer<dtype>(std::move(dim.total()), init));
        }
    }

    /**
     * @brief Fill internal buffer with given value
     */
    void fill(dtype init) {
        if (m_type == buffer_data) {
            std::get<shared_buffer_ptr>(m_data)->fill(init);
        } else {
            std::get<shared_cu_ptr>(m_data)->fill(init);
        }
    }

    /**
     * @brief Apply given generator to each element of the tensor
     */
    void generate(std::function<dtype()> gen) {
        if (m_type == buffer_data) {
            std::get<shared_buffer_ptr>(m_data)->generate(gen);
        } else {
            std::get<shared_cu_ptr>(m_data)->generate(gen);
        }
    }
private:
    variant_data m_data{shared_buffer_ptr<dtype>(nullptr)};
    internal_type m_type{buffer_data};
};

END_BLUST_NAMESPACE