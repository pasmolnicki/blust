#pragma once

#include "namespaces.hpp"

#include <stdexcept>

START_BLUST_NAMESPACE

class InvalidMatrixSize : public std::runtime_error
{
public:
    InvalidMatrixSize() : std::runtime_error("Invalid matrix size") {}

    /**
     * @brief Create expection object with matrix shape
     * @brief got shape of the matrix that we got (first = row, second = cols)
     * @brief expected shape
     */
    InvalidMatrixSize(std::pair<size_t, size_t> got, std::pair<size_t, size_t> expected) :
        std::runtime_error(
            "Got matrix: r=" + std::to_string(got.first) + " c=" + std::to_string(got.second) +
            ", expected: r=" + std::to_string(expected.first) + " c=" + std::to_string(expected.second)
        ) {}
};

class AssertError : public std::runtime_error
{
public:
    AssertError(const char* file, int line, const char* condition) :
        std::runtime_error(std::string("Assertion failed: ") + condition + " at " + file + ":" + std::to_string(line)) {}
};

// Throw an exception if the condition is not met
inline void throw_assert(const char* file, int line, const char* condition)
{
    throw AssertError(file, line, condition);
}

// Assert for easier debugging
#define BLUST_ASSERT(cond) cond ? (void)0 : throw_assert(__FILE__, __LINE__, #cond)

END_BLUST_NAMESPACE