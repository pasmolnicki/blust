#pragma once

#include <blust/namespaces.hpp>
#include <blust/base_types.hpp>
#include <blust/utils.hpp>
#include <blust/error.hpp>
#include <blust/backend/optimized.hpp>

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
    friend class cpu_backend;

    static_assert(std::is_arithmetic<dtype>(), 
        "Template parameter in matrix must be an arithmetic type (int, float, double, etc.)");

    typedef dtype* pointer_t;
    typedef const dtype* const_pointer_t;

    matrix() : m_rows(0), m_cols(0) {}
    matrix(const matrix& other) { *this = other; }
    matrix(matrix&& other) noexcept { *this = std::forward<matrix>(other); }

    /**
     * @brief Create a matrix (row x col) size, filled with `init_val`
     * @param shape shape of the matrix...
     * @param init_val value to fill up the matrix with
     */
    matrix(shape2D shape, dtype init_val = 0)
    {
        build(shape, init_val);
    }

    // build from initializer list
    matrix(std::initializer_list<std::vector<dtype>> list)
    {
        size_t cols = list.begin()->size();
        M_alloc_buffer({list.size(), list.begin()->size()});

        size_t r = 0;
        for(auto& v : list)
        {
            // invalid column size
            if (v.size() != cols)
                throw InvalidMatrixSize();

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

    constexpr matrix& operator=(matrix&& other) noexcept
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
    inline size_t size() const noexcept { return m_matrix.size(); }

    // Get number of rows in a matrix
    inline size_t rows() const noexcept { return m_rows; }

    // Get number of columns
    inline size_t cols() const noexcept { return m_cols; }

    // Get dimensions of a matrix
    inline shape2D dim() const noexcept { return {rows(), cols()}; }

    // Get the raw pointer
    inline const_pointer_t data() const { return m_matrix.data(); }
    inline pointer_t data() { return m_matrix.data(); }

    inline auto begin() { return m_matrix.begin(); }
    inline auto begin() const { return m_matrix.begin(); }
    inline auto end() { return m_matrix.end(); }
    inline auto end() const { return m_matrix.end(); }

    // Get transposed matrix (GPU)
    matrix T()
    {
        matrix m({m_cols, m_rows});
		g_backend->mat_transpose(m.data(), data(), m_rows, m_cols);
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
    friend matrix<dtype> operator+(matrix<dtype>& lhs, matrix<dtype>& rhs)
    {
        matrix<dtype> ret(lhs);
        ret.M_helper_add_m(ret, rhs);
        return ret;
    }

    // Add 2 matrices
    friend matrix<dtype> operator+(matrix<dtype>&& lhs, matrix<dtype>& rhs)
    {
        matrix<dtype> ret(std::forward<matrix<dtype>>(lhs));
        ret.M_helper_add_m(ret, rhs);
        return ret;
    }

    // Add 2 matrices
    friend matrix<dtype> operator+(matrix<dtype>& lhs, matrix<dtype>&& rhs)
    {
        matrix<dtype> ret(lhs);
        ret.M_helper_add_m(ret, rhs);
        return ret;
    }

    // Add given matrix to this one
    matrix& operator+=(matrix<dtype>& m)
    {
        M_helper_add_m(*this, m);
        return *this;
    }

    // Add given matrix to this one
    matrix& operator+=(matrix<dtype>&& m)
    {
        M_helper_add_m(*this, m);
        return *this;
    }

    // Substract 2 matrices
    friend matrix<dtype> operator-(matrix<dtype>& lhs, matrix<dtype>& rhs)
    {
        matrix<dtype> ret(lhs);
        ret.M_helper_sub_m(ret, rhs);
        return ret;
    }

    // Substract 2 matrices
    friend matrix<dtype> operator-(matrix<dtype>&& lhs, matrix<dtype>& rhs)
    {
        matrix<dtype> ret(std::forward<matrix<dtype>>(lhs));
        ret.M_helper_sub_m(ret, rhs);
        return ret;
    }

    friend matrix<dtype> operator-(matrix<dtype>& lhs, matrix<dtype>&& rhs)
    {
        matrix<dtype> ret(lhs);
        ret.M_helper_sub_m(ret, rhs);
        return ret;
    }

    // Substract given matrix to this one
    matrix& operator-=(matrix<dtype>& m)
    {
        M_helper_sub_m(*this, m);
        return *this;
    }

    // Substract given matrix to this one
    matrix& operator-=(matrix<dtype>&& m)
    {
        M_helper_sub_m(*this, m);
        return *this;
    }

    // Multiply 2 matrices
    matrix matmul(matrix& m) { return M_multip(m); }

    // Multiplication of 2 matrices
    friend matrix<dtype> operator*(matrix<dtype>& lhs, matrix<dtype>& rhs) { return lhs.M_multip(rhs); }
    friend matrix<dtype> operator*(matrix<dtype>&& lhs, matrix<dtype>& rhs) { return lhs.M_multip(rhs); }
    friend matrix<dtype> operator*(matrix<dtype>& lhs, matrix<dtype>&& rhs) { return lhs.M_multip(rhs); }
    friend matrix<dtype> operator*(matrix<dtype>&& lhs, matrix<dtype>&& rhs) { return lhs.M_multip(rhs); }

    // Multiply matricies with hadamard product (Cij = (Aij * Bij))
    friend matrix<dtype> operator%(matrix<dtype>& lhs, matrix<dtype>& rhs) 
    {
        matrix<dtype> m(lhs);
        m.M_helper_hadamard_mul(m, rhs);
        return m;
    }

    // Multiply matricies with hadamard product (Cij = (Aij * Bij))
    friend matrix<dtype> operator%(matrix<dtype>&& lhs, matrix<dtype>& rhs)
    {
        matrix<dtype> m(std::forward<matrix<dtype>>(lhs));
        m.M_helper_hadamard_mul(m, rhs);
        return m;
    }

    friend matrix<dtype> operator%(matrix<dtype>& lhs, matrix<dtype>&& rhs)
    {
        matrix<dtype> m(lhs);
        m.M_helper_hadamard_mul(m, rhs);
        return m;
    }

    friend matrix<dtype> operator%(matrix<dtype>&& lhs, matrix<dtype>&& rhs)
    {
        matrix<dtype> m(std::forward<matrix<dtype>>(lhs));
        m.M_helper_hadamard_mul(m, rhs);
        return m;
    }

    matrix& operator%=(matrix<dtype>& m)
    {
        M_helper_hadamard_mul(*this, m);
        return *this;
    }

    matrix& operator%=(matrix<dtype>&& m)
    {
        M_helper_hadamard_mul(*this, m);
        return *this;
    }

    // Multiply matricies with hadamard product (Cij = (Aij * Bij)) (same as `A % B`)
    matrix hadamard(matrix<dtype>& mul)
    {
        matrix<dtype> m(*this);
        m.M_helper_hadamard_mul(m, mul);
        return m;
    }

    // Multiply matricies with hadamard product (Cij = (Aij * Bij)) (same as `A % B`)
    matrix hadamard(matrix<dtype>&& mul)
    {
        matrix<dtype> m(std::forward<decltype(mul)>(mul)); // don't copy the buffer
        m.M_helper_hadamard_mul(m, *this);
        return m;
    }

    // Multiply this matrix with `mul`, and set the result as this
    matrix& operator*=(matrix<dtype>& mul)
    {
        *this = M_multip(mul);
        return *this;
    }

    // Multiply matrix by a scalar
    friend matrix<dtype> operator*(matrix<dtype>& lhs, number_t d) { return lhs.M_multip_k(d); }
    friend matrix<dtype> operator*(number_t d, matrix<dtype>& rhs) { return rhs.M_multip_k(d); }

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
        m_matrix.resize(m_cols * m_rows, init);
    }

    // Assert m1 and m2 are equally shaped
    static void M_assert_eq_dim(matrix<dtype>& m1, matrix<dtype>& m2)
    {
        if (m1.dim() != m2.dim())
            throw InvalidMatrixSize({m2.rows(), m2.cols()}, {m1.rows(), m1.cols()});
    }

    // Assert m1 and m2 can be multiplied (m1.cols == m2.rows)
    static void M_assert_dim_mul(matrix<dtype>& m1, matrix<dtype>& m2)
    {
        if (!(m1.cols() == m2.rows()))
            throw InvalidMatrixSize({m2.rows(), m2.cols()}, {m1.cols(), m2.cols()});
    }

    // Add matrix `m2` to `m1` (result stored in m1) (GPU)
    static void M_helper_add_m(matrix<dtype>& m1, matrix<dtype>& m2)
    {
        M_assert_eq_dim(m1, m2);
		g_backend->vector_add(m1.data(), m1.data(), m2.data(), m1.size());
    }

    // Substract matrix `m1` from `m2` (result is stored in m1) (GPU)
    static void M_helper_sub_m(matrix<dtype>& m1, matrix<dtype>& m2)
    {
        M_assert_eq_dim(m1, m2);
		g_backend->vector_sub(m1.data(), m1.data(), m2.data(), m1.size());
    }

    // Perform hadamard multiplication (Cij = Aij * Bij), store the result in `m1` (GPU)
    static void M_helper_hadamard_mul(matrix<dtype>& m1, matrix<dtype>& m2)
    {
        M_assert_eq_dim(m1, m2);
		g_backend->vector_mul_hadamard(m1.data(), m1.data(), m2.data(), m1.size());
    }

    /**
     * @brief Multiply matrices. (GPU)
     * @throw May throw `InvalidMatrixSize` if this->cols() != m.rows()
     * @return Product matix (rows() x m.cols())
     */
    matrix M_multip(matrix& m)
    {
        M_assert_dim_mul(*this, m);        
        matrix ret({rows(), m.cols()});
		g_backend->mat_mul(ret.data(), this->data(), m.data(), m_rows, m.cols(), m.rows());
        return ret;
    }

    // Multiply the matrix by a scalar (GPU)
    matrix M_multip_k(number_t k)
    {
        matrix m = *this;
		g_backend->vector_scalar_mul(m.data(), m.data(), k, m.size());
        return m;
    }
};

END_BLUST_NAMESPACE