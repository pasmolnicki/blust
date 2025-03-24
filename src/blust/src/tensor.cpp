#include <blust/tensor.hpp>


START_BLUST_NAMESPACE

int tensor::n_allocs = 0;
int tensor::max_allocs = 0;


void tensor::M_print_tensor(
    const tensor& t, std::ostream& out, size_t rank, 
    size_t index, size_t offset
) noexcept
{
    auto end = t.m_shape.m_dims.at(index);

    // Text tabluation
    for (size_t p = 0; p < index; p++)
        out << ' ';
    
    // Got 1D representation
    if (rank == 1)
    {            
        out << '[';
        for (size_t i = 0; i < end; i++)
        {
            // print with proper formatting
            if (i == end - 1)
                out << t.m_tensor.data[offset + i];
            else
                out << t.m_tensor.data[offset + i] << ", ";   
        }
    }
    else
    {
        out << "[\n";
        for (size_t i = 0; i < end; i++)
        {
            // Go to next dimension of the tensor
            M_print_tensor(t, out, rank - 1, index + 1, offset + i * t.m_shape.m_dims[index + 1]);
        }

        // Text tabluation
        for (size_t p = 0; p < index; p++)
            out << ' ';
    }


    if (rank != t.rank())
        out << "],\n";
    else
        out << "]\n";
}

tensor& tensor::operator=(const tensor& t) noexcept
{
    M_cleanup_buffer();
    auto count = M_alloc_buffer(t);

    if (count == 0)
        return *this;

    // If that's a cuda pointer, memcpy to this buffer
    if (t.m_data_type == data_type::cuda && t.m_tensor.cu_ptr != 0) {
        cuMemcpyDtoH(
            m_tensor.data, t.m_tensor.cu_ptr, 
            count * sizeof(number_t));
    }
    else
    {
        std::copy_n(t.m_tensor.data, count, m_tensor.data); // will memcpy the buffer
    }

    return *this;
}

tensor& tensor::operator=(tensor&& t) noexcept
{
    m_bytesize  = t.m_bytesize;
    m_shape     = std::forward<shape>(t.m_shape);
    m_data_type = data_type::buffer;
    m_shared    = false; // since I'm getting the ownership of the pointer

    // If that's a cuda pointer, copy the buffer
    if (t.m_data_type == data_type::cuda && t.m_tensor.cu_ptr != 0)
    {
        const auto count    = size();
        m_tensor.data       = aligned_alloc(count);
        cuMemcpyDtoH(
            m_tensor.data, t.m_tensor.cu_ptr, 
            count * sizeof(number_t));
    }
    else
    {
        // Just release the buffer
        M_cleanup_buffer();
        m_tensor.data = t.release();
    }

    return *this;
}

inline std::ostream& operator<<(std::ostream& out, const tensor& t) noexcept
{
    out << "<tensor: dtype=" << utils::TypeName<number_t>() << " " << t.m_shape << ">\n";

    // print the buffer
    if (t.m_data_type == tensor::data_type::buffer)
    {
        auto rank = t.rank();
        if (rank >= 1)
            t.M_print_tensor(t, out, rank);
    }
    
    return out;
}

/**
 * @brief Allocates the buffer with the same size as `t`, copies the dimension, and sets the 
 * data type to buffer, but DOES NOT COPY THE CONTENT of t's data
 * @returns t.size() (if 0 then buffer was not allocated and is set to `nullptr`)
 */
inline size_t tensor::M_alloc_buffer(const tensor& t) noexcept
{
    const auto count    = t.size();
    m_shape             = t.m_shape;
    m_shared            = false;
    m_data_type         = data_type::buffer; // always use buffer
    m_tensor.data       = nullptr;
    m_bytesize          = 0;
    
    if (count == 0) 
        return 0;

    inc_alloc(1);
    m_bytesize          = get_bytesize(count);
    m_tensor.data       = aligned_alloc(count);
    return count;
}

END_BLUST_NAMESPACE