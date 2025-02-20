#pragma once

#include <blust/namespaces.hpp>
#include <blust/utils.hpp>
#include <blust/error.hpp>

#include <vector>
#include <iostream>
#include <memory>

START_BLUST_NAMESPACE

class shape2D
{
public:
    size_t x, y;
    shape2D() : x(0), y(0) {}

    /**
     * @brief Set the shape, (y, x) (rows, columns, in matrix)
     */
    shape2D(size_t x, size_t y) : x(x), y(y) {}
    shape2D(const shape2D& other) {*this = other;}

    shape2D& operator=(const shape2D& other) 
    {
        this->x = other.x; this->y = other.y;
        return *this;
    }

    friend bool operator==(const shape2D& lhs, const shape2D& rhs) {
        return lhs.x == rhs.x && lhs.y == rhs.y;
    }

    friend bool operator!=(const shape2D& lhs, const shape2D& rhs) {
        return !(lhs == rhs);
    }

    // Operator for printing to output stream
    friend std::ostream& operator<<(std::ostream& out, const shape2D& shape) {
        return out << shape.x << 'x' << shape.y;
    }
};


// Matrix with given `dtype` 
template <typename dtype>
class matrix
{
public:
    static_assert(std::is_arithmetic<dtype>(), 
        "Template parameter in matrix must be an arithmetic type (int, float, double, etc.)");

    typedef dtype* pointer_t;
    typedef const dtype* const_pointer_t;

    matrix() = default;
    matrix(const matrix& other) { *this = other; }
    matrix(matrix&& other) { *this = std::forward<matrix>(other); }

    /**
     * @brief Create a matrix (row x col) size, filled with `init_val`
     * @param shape shape of the matrix...
     * @param init_val value to fill up the matrix with
     */
    matrix(shape2D shape, dtype init_val = 0)
    {
        build(shape, init_val);
    }

    // Setup 
    matrix(std::initializer_list<std::vector<dtype>> list)
    {
        M_alloc_buffer({list.size(), list.begin()->size()});

        size_t r = 0;
        for(auto& v : list)
        {
            for (size_t i = 0; i < v.size(); i++)
                (*this)(r, i) = v[i];
            r++;
        }
    }

    // Create matrix from a flat vector
    matrix(shape2D shape, std::vector<dtype>& v)
    {
        m_rows   = shape.x;
        m_cols   = shape.y;
        m_matrix = std::move(v);
    }

    matrix& operator=(matrix&& other)
    {
        m_rows   = other.m_rows; 
        m_cols   = other.m_cols;
        m_matrix = std::move(other.m_matrix);
        return *this;
    }

    matrix& operator=(const matrix& other)
    {
        return (this->operator=<dtype>(other));
    }

    /**
     * @brief Copy operator
     */
    template <typename T>
    matrix& operator=(const matrix<T>& other)
    {
        m_rows   = other.m_rows;
        m_cols   = other.m_cols;
        m_matrix = other.m_matrix;
        return *this;
    }

    /**
     * @brief Create matrix R x C
     * @param r n rows
     * @param c n cols
     */
    inline void build(shape2D shape, dtype init = 0)
    {
        M_alloc_buffer(shape, init);
    }

    // Get the total size of the buffer
    inline size_t size() const { return m_rows * m_cols; }

    // Get number of rows in a matrix
    inline size_t rows() const { return m_rows; }

    // Get number of columns
    inline size_t cols() const { return m_cols; }

    // Get dimensions of a matrix
    inline shape2D dim() const { return {rows(), cols()}; }

    // Get the raw pointer
    inline const_pointer_t data() const { return m_matrix.data(); }
    inline pointer_t data() { return m_matrix.data(); }

    inline auto begin() { return m_matrix.begin(); }
    inline auto begin() const { return m_matrix.begin(); }
    inline auto end() { return m_matrix.end(); }
    inline auto end() const { return m_matrix.end(); }

    // Get transposed matrix 
    matrix T()
    {
        matrix m({m_cols, m_rows});
        const auto s = size();
        for (size_t n = 0; n < s; ++n)
        {
            int i = n / m_rows;
            int j = n % m_rows;
            m.m_matrix[n] = m_matrix[m_cols*j + i];
        }
        return m;
    }

    // Get the value at (row, column)
    dtype& operator()(size_t r, size_t c) { return m_matrix[r * m_cols + c]; }
    const dtype& operator()(size_t r, size_t c) const { return m_matrix[r * m_cols + c]; }

    // Get value at index i, (assumes index is correct)
    dtype& operator()(size_t i) { return m_matrix[i]; }
    const dtype& operator()(size_t i) const { return m_matrix[i]; }

    // Get the whole row as vector
    std::vector<dtype> operator[](size_t r) const
    { 
        return std::vector<dtype>(
            m_matrix.begin() + (r * m_cols), 
            m_matrix.begin() + ((r + 1) * m_cols)); 
    }

    // Compare the matrices
    template <typename t>
    friend inline bool operator==(const matrix& rhs, const matrix<t>& lhs)
    {
        if (!(rhs.rows() == lhs.rows() && rhs.cols() == lhs.cols()))
            return false;
        
        auto rb = lhs.data(), 
             re = lhs.data() + lhs.size();
        auto lb = rhs.data();
        return std::equal(rb, re, lb);
    }

    // Add 2 matrices
    template <typename T>
    friend matrix<dtype> operator+(matrix<dtype>& lhs, matrix<T>& rhs) 
    {
        matrix<dtype> ret(lhs);
        ret.M_helper_add_m(ret, rhs);
        return ret;
    }

    // Add 2 matrices
    template <typename T>
    friend matrix<dtype> operator+(matrix<dtype>&& lhs, matrix<T>& rhs) 
    {
        matrix<dtype> ret(std::forward<matrix<dtype>>(lhs));
        ret.M_helper_add_m(ret, rhs);
        return ret;
    }

    // Add 2 matrices
    template <typename T>
    friend matrix<dtype> operator+(matrix<dtype>& lhs, matrix<T>&& rhs) 
    {
        matrix<dtype> ret(lhs);
        ret.M_helper_add_m(ret, rhs);
        return ret;
    }

    // Add given matrix to this one
    template <typename T>
    matrix& operator+=(matrix<T>& m) 
    {
        M_helper_add_m(*this, m);
        return *this;
    }

    // Add given matrix to this one
    template <typename T>
    matrix& operator+=(matrix<T>&& m) 
    {
        M_helper_add_m(*this, m);
        return *this;
    }

    // Substract 2 matrices
    template <typename T>
    friend matrix<dtype> operator-(matrix<dtype>& lhs, matrix<T>& rhs) 
    {
        matrix<dtype> ret(lhs);
        ret.M_helper_sub_m(ret, rhs);
        return ret;
    }

    // Substract 2 matrices
    template <typename T>
    friend matrix<dtype> operator-(matrix<dtype>&& lhs, matrix<T>& rhs) 
    {
        matrix<dtype> ret(std::forward<matrix<dtype>>(lhs));
        ret.M_helper_sub_m(ret, rhs);
        return ret;
    }

    template <typename T>
    friend matrix<dtype> operator-(matrix<dtype>& lhs, matrix<T>&& rhs) 
    {
        matrix<dtype> ret(lhs);
        ret.M_helper_sub_m(ret, rhs);
        return ret;
    }

    // Substract given matrix to this one
    template <typename T>
    matrix& operator-=(matrix<T>& m) 
    {
        M_helper_sub_m(*this, m);
        return *this;
    }

    // Substract given matrix to this one
    template <typename T>
    matrix& operator-=(matrix<T>&& m) 
    {
        M_helper_sub_m(*this, m);
        return *this;
    }

    // Multiply 2 matrices
    template <typename T>
    matrix matmul(matrix<T>& m) { return M_multip(m); }

    // Multiplication of 2 matrices
    template <typename T>
    friend matrix<dtype> operator*(matrix<dtype>& lhs, matrix<T>& rhs) { return lhs.M_multip(rhs); }

    template <typename T>
    friend matrix<dtype> operator*(matrix<dtype>&& lhs, matrix<T>& rhs) { return lhs.M_multip(rhs); }

    template <typename T>
    friend matrix<dtype> operator*(matrix<dtype>& lhs, matrix<T>&& rhs) { return lhs.M_multip(rhs); }

    template <typename T>
    friend matrix<dtype> operator*(matrix<dtype>&& lhs, matrix<T>&& rhs) { return lhs.M_multip(rhs); }

    // Multiply matricies with hadamard product (Cij = (Aij * Bij))
    template <typename T>
    friend matrix<dtype> operator%(matrix<dtype>& lhs, matrix<T>& rhs) 
    {
        matrix<dtype> m(lhs);
        m.M_helper_hadamard_mul(m, rhs);
        return m;
    }

    // Multiply matricies with hadamard product (Cij = (Aij * Bij))
    template <typename T>
    friend matrix<dtype> operator%(matrix<dtype>&& lhs, matrix<T>& rhs) 
    {
        matrix<dtype> m(std::forward<matrix<dtype>>(lhs));
        m.M_helper_hadamard_mul(m, rhs);
        return m;
    }

    template <typename T>
    friend matrix<dtype> operator%(matrix<dtype>& lhs, matrix<T>&& rhs) 
    {
        matrix<dtype> m(lhs);
        m.M_helper_hadamard_mul(m, rhs);
        return m;
    }

    template <typename T>
    friend matrix<dtype> operator%(matrix<dtype>&& lhs, matrix<T>&& rhs) 
    {
        matrix<dtype> m(std::forward<matrix<dtype>>(lhs));
        m.M_helper_hadamard_mul(m, rhs);
        return m;
    }

    template <typename T>
    matrix& operator%=(matrix<T>& m)
    {
        M_helper_hadamard_mul(*this, m);
        return *this;
    }

    template <typename T>
    matrix& operator%=(matrix<T>&& m)
    {
        M_helper_hadamard_mul(*this, m);
        return *this;
    }

    // Multiply matricies with hadamard product (Cij = (Aij * Bij)) (same as `A % B`)
    template <typename T>
    matrix hadamard(matrix<T>& mul) 
    {
        matrix<dtype> m(*this);
        m.M_helper_hadamard_mul(m, mul);
        return m;
    }


    // Multiply this matrix with `mul`, and set the result as this
    template <typename T>
    matrix& operator*=(matrix<T>& mul) 
    {
        *this = M_multip(mul);
        return *this;
    }

    // Multiplication of matrix and vector (for simplification, vector is used as if it was vertical)
    // Resulting in vector (also vertical), of size matrix.rows (should be a matrix of dimensions: matrix.rows x 1) 
    template <typename T>
    friend std::vector<dtype> operator*(matrix<dtype>& lhs, std::vector<T>& rhs) { return lhs.M_multip_v<T, true>(rhs); }

    // Multiply vector (1d matrix) by a matrix
    template <typename T>
    friend std::vector<dtype> operator*(std::vector<T>& lhs, matrix<dtype>& rhs) { return rhs.M_multip_v<T, false>(lhs); }

    // Multiply matrix by a scalar
    friend matrix<dtype> operator*(matrix<dtype>& lhs, double d) { return lhs.M_multip_k(d); }
    friend matrix<dtype> operator*(double d, matrix<dtype>& rhs) { return rhs.M_multip_k(d); }
    friend matrix<dtype> operator*(matrix<dtype>& lhs, int d) { return lhs.M_multip_k(d); }
    friend matrix<dtype> operator*(int d, matrix<dtype>& rhs) { return rhs.M_multip_k(d); }

    // Print the matrix to output stream
    friend std::ostream& operator<<(std::ostream& out, const matrix& m)
    {
        out << "<dtype=" << utils::TypeName<dtype>() << ", dim=" << m.dim() << ">\n";
        for (size_t r = 0; r < m.rows(); ++r)
        {
            out << '[';
            for (size_t c = 0; c < m.cols(); ++c)
            {
                // compare the addresses 
                if (c == m.cols() - 1)
                    out << m(r, c);
                else
                    out << m(r, c) << ", ";
            }
            out << "]\n";
        }
        
        return out;
    }

private:

    std::vector<dtype> m_matrix;
    size_t m_rows;
    size_t m_cols;

    // Set the internal size and reallocate the buffer
    void M_alloc_buffer(shape2D shape, dtype init = 0)
    {
        m_rows = shape.x;
        m_cols = shape.y;
        m_matrix.resize(size(), init);
    }

    // Assert m1 and m2 are equally shaped
    template <typename T>
    static void M_assert_eq_dim(matrix<dtype>& m1, matrix<T>& m2) 
    {
        if (m1.dim() != m2.dim())
            throw InvalidMatrixSize({m2.rows(), m2.cols()}, {m1.rows(), m1.cols()});
    }

    // Assert m1 and m2 can be multiplied (m1.cols == m2.rows)
    template <typename T>
    static void M_assert_dim_mul(matrix<dtype>& m1, matrix<T>& m2) 
    {
        if (!(m1.cols() == m2.rows()))
            throw InvalidMatrixSize({m2.rows(), m2.cols()}, {m1.cols(), m2.cols()});
    }

    // Add matrix `m2` to `m1` (result stored in m1)
    template <typename T>
    static void M_helper_add_m(matrix<dtype>& m1, matrix<T>& m2)
    {
        M_assert_eq_dim(m1, m2);
        
        const auto size = m1.size();
        for (size_t i = 0; i < size; ++i) {
            m1.m_matrix[i] += m2.m_matrix[i];
        }
    }

    // Substract matrix `m1` from `m2` (result is stored in m1)
    template <typename T>
    static void M_helper_sub_m(matrix<dtype>& m1, matrix<T>& m2)
    {
        M_assert_eq_dim(m1, m2);
        
        const auto size = m1.size();
        for (size_t i = 0; i < size; ++i) {
            m1.m_matrix[i] -= m2.m_matrix[i];
        }
    }

    // Perform hadamard multiplication (Cij = Aij * Bij), store the result in `m1`
    template <typename T>
    static void M_helper_hadamard_mul(matrix<dtype>& m1, matrix<T>& m2)
    {
        M_assert_eq_dim(m1, m2);

        const auto size = m1.size();
        for (size_t i = 0; i < size; ++i) {
            m1.m_matrix[i] *= m2.m_matrix[i];
        }
    }

    // dot product of given vectors, assumes the input is correct (v1.size == v2.size)
    template<typename T>
    dtype M_dot_product(std::vector<dtype>& v1, std::vector<T>& v2)
    {
        const size_t n = v1.size();
        dtype dot      = 0;
        size_t i       = 0;

        // Unrolled
        if (n >= 4)
        {
            for (i = 0; i <= n - 4; i += 4)
            {
                dot += (v1[i]     * v2[i] +
                        v1[i + 1] * v2[i + 1] +
                        v1[i + 2] * v2[i + 2] +
                        v1[i + 3] * v2[i + 3]
                );
            }
        }
        
        for (; i < n; i++)
            dot += v1[i] * v2[i];

        return dot;
    }

    // Optimized vector multiplication
    template <typename T, bool MatrixFirst>
    std::vector<dtype> M_multip_v(std::vector<T>& v)
    {
        if constexpr (MatrixFirst)
        {
            // M * v, to make sense out of this, v is assumed to be vertical
            // Assert correct sizes
            if (!(cols() == v.size()))
                throw InvalidMatrixSize({1, v.size()}, {1, cols()});
        
            const size_t n_rows = rows();
            std::vector<dtype> result(n_rows, 0);

            // Calculate the dot product for each row
            for (size_t r = 0; r < n_rows; r++)
            {
                auto row  = (*this)[r];
                result[r] = M_dot_product(row, v);
            }
            return result;
        }
        else
        {
            // v * M
            if (!(v.size() == rows()))
                throw InvalidMatrixSize({v.size(), 1}, {rows(), 1});
            
            const size_t n_cols = cols();
            std::vector<dtype> result(n_cols, 0);

            // Get the transposed matrix, for easier memory access
            auto transp = this->T();

            // Calculate the dot product for each row
            for (size_t c = 0; c < n_cols; c++)
            {
                auto col  = transp[c];
                result[c] = M_dot_product(col, v);
            }
            return result;
        }
    }

    /**
     * @brief Multiply matrices.
     * @throw May throw `InvalidMatrixSize` if this->cols() != m.rows()
     * @return Product matix (rows() x m.cols())
     */
    template <typename t>
    matrix M_multip(matrix<t>& m)
    {
        M_assert_dim_mul(*this, m);
        
        const size_t m_rows = m.rows(),
                     m_cols = m.cols(),
                     n_rows = rows();
        
        matrix ret({n_rows, m_cols});

        // re-order
        for (size_t r1 = 0; r1 < n_rows; r1++) // go through the rows of 1st matrix
            for(size_t k = 0; k < m_rows; ++k) // reorder, go through the rows of 2nd matrix
                for(size_t c2 = 0; c2 < m_cols; c2++) // loop through the columns of 2nd matrix
                    ret(r1, c2) += (*this)(r1, k) * (m(k, c2)); // dot product
        
        return ret;
    }

    // Multiply the matrix by a scalar
    template <typename t>
    matrix M_multip_k(t k)
    {
        static_assert(std::is_arithmetic<t>(), 
            "Given type must be arithmetic (int, double, float, etc.)");

        matrix m = *this;

        for (size_t r = 0; r < rows(); r++)
            for (size_t c = 0; c < cols(); c++)
                m(r, c) *= k;
            
        return m;
    }
};

END_BLUST_NAMESPACE