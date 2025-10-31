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
        auto data = std::get<pointer>(t.m_tensor);
        out << '[';
        for (size_t i = 0; i < end; i++)
        {
            // print with proper formatting
            if (i == end - 1)
                out << data[offset + i];
            else
                out << data[offset + i] << ", ";
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
    if (t.m_data_type == pointer_type::cuda && t.cu_data() != 0) {
        cuMemcpyDtoH(
            data(), t.cu_data(), 
            count * sizeof(number_t));
    }
    else
    {
        std::copy_n(t.data(), count, data()); // will memcpy the buffer
    }

    return *this;
}

tensor& tensor::operator=(tensor&& t) noexcept
{
    m_bytesize  = t.m_bytesize;
    m_shape     = std::forward<shape>(t.m_shape);
    m_data_type = pointer_type::buffer;
    m_shared    = false; // since I'm getting the ownership of the pointer

    // If that's a cuda pointer, copy the buffer
    if (t.m_data_type == pointer_type::cuda && std::holds_alternative<cu_pointer>(m_tensor))
    {
        const auto count    = size();
        m_tensor            = aligned_alloc(count);
        cuMemcpyDtoH(
            data(), t.cu_data(), 
            count * sizeof(number_t));
    }
    else
    {
        // Just release the buffer
        M_cleanup_buffer();
        m_tensor = t.release();
    }

    return *this;
}

std::ostream& operator<<(std::ostream& out, const tensor& t) noexcept
{
    out << "<tensor: dtype=" << utils::TypeName<number_t>() << " " << t.m_shape << ">\n";
    auto prev = out.precision(2);
    out << std::fixed;

    // print the buffer
    if (t.m_data_type == tensor::pointer_type::buffer)
    {
        auto rank = t.rank();
        if (rank >= 1)
            t.M_print_tensor(t, out, rank);
    }
    
    out.precision(prev);
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
    m_data_type         = pointer_type::buffer; // always use buffer
    m_tensor            = (pointer)nullptr;
    m_bytesize          = 0;
    
    if (count == 0) 
        return 0;

    inc_alloc(1);
    m_bytesize          = get_bytesize(count);
    m_tensor            = aligned_alloc(count);
    return count;
}

END_BLUST_NAMESPACE