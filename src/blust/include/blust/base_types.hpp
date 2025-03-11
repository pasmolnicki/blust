#pragma once

#include "namespaces.hpp"

#include <vector>
#include <ostream>

START_BLUST_NAMESPACE

typedef float number_t;
typedef std::vector<number_t> vector_t;

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

END_BLUST_NAMESPACE