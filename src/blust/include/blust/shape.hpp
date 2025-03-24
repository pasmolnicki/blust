#pragma once

#include <numeric>

#include "base_types.hpp"

START_BLUST_NAMESPACE

// Shape of the tensor
class shape
{
public:
    friend class tensor;

    typedef std::vector<size_t> dim_t;

    // Create a shape object with given inital dimensions as a initializer list
    shape(std::initializer_list<int> dims) noexcept
    {
        m_dims.reserve(dims.size());
        for (auto& d : dims)
            m_dims.push_back(d);
    }

    // c'tors
    shape() = default;
    shape(const shape& other) noexcept : m_dims(other.m_dims) {}
    shape(shape&& other) noexcept : m_dims(std::forward<dim_t>(other.m_dims)) {}
    
    // Copy operator
    shape& operator=(const shape& other) noexcept
    { 
        m_dims = other.m_dims; 
        return *this; 
    }

    // Move operator
    shape& operator=(shape&& other) noexcept
    {
        m_dims = std::forward<dim_t>(other.m_dims);
        return *this;
    }

    // Get the vector of dimensions
    const dim_t& dim() const noexcept { return m_dims; }

    // Get the rank of the tensor ( == dim.size)
    size_t rank() const noexcept { return m_dims.size(); }

    // Get total number of elements
    size_t total() const noexcept {
        return m_dims.empty() ? 0 : std::accumulate(m_dims.begin(), m_dims.end(), 1, std::multiplies<size_t>());
    }

    // Clear the shape
    void clear() noexcept { m_dims.clear(); }

    // Printing shape
    friend std::ostream& operator<<(std::ostream& out, const shape& s) noexcept
    {
        if (s.rank() == 0) {
            out << "shape=none";
            return out;
        }

        const auto rank = s.rank();
        out << "rank=" << rank << " dim=";
        for (size_t i = 0; i < rank; i++)
        {
            out << s.m_dims[i];
            if (i != rank - 1)
                out << 'x';
        }
        return out;
    }

private:
    dim_t m_dims;
};

END_BLUST_NAMESPACE