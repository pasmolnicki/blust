#pragma once

#include "base_types.hpp"

#include <random>
#include <algorithm>
#include <array>
#include <cstddef>
#include <string_view>
#include <iostream>


namespace std
{
    // Vector printing
    template <typename T>
    std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
    {
        auto last = v.end() - 1;
        os << '[';
        for (auto& e : v)
        {
            // compare the addresses 
            if (&e == &*last)
                os << e;
            else
                os << e << ", ";
        }
        os << ']';

        return os;
    }
}

START_BLUST_NAMESPACE
namespace utils
{
    
    // Source: https://stackoverflow.com/posts/59522794/revisions
    // Author: HolyBlackCat
    namespace type_names
    {
        template <typename T>
        constexpr const auto &RawTypeName()
        {
            #ifdef _MSC_VER
            return __FUNCSIG__;
            #else
            return __PRETTY_FUNCTION__;
            #endif
        }
    
        struct RawTypeNameFormat
        {
            std::size_t leading_junk = 0, trailing_junk = 0;
        };
    
        // Returns `false` on failure.
        inline constexpr bool GetRawTypeNameFormat(RawTypeNameFormat *format)
        {
            const auto &str = RawTypeName<int>();
            for (std::size_t i = 0;; i++)
            {
                if (str[i] == 'i' && str[i+1] == 'n' && str[i+2] == 't')
                {
                    if (format)
                    {
                        format->leading_junk = i;
                        format->trailing_junk = sizeof(str)-i-3-1; // `3` is the length of "int", `1` is the space for the null terminator.
                    }
                    return true;
                }
            }
            return false;
        }
    
        inline static constexpr RawTypeNameFormat format =
        []{
            static_assert(GetRawTypeNameFormat(nullptr), "Unable to figure out how to generate type names on this compiler.");
            RawTypeNameFormat format;
            GetRawTypeNameFormat(&format);
            return format;
        }();

        // Returns the type name in a `std::array<char, N>` (null-terminated).
        template <typename T>
        constexpr auto CexprTypeName()
        {
            constexpr std::size_t len = 
                sizeof(type_names::RawTypeName<T>()) 
                - type_names::format.leading_junk 
                - type_names::format.trailing_junk;
            
            std::array<char, len> name{};
            for (std::size_t i = 0; i < len-1; i++)
                name[i] = type_names::RawTypeName<T>()[i + type_names::format.leading_junk];
            return name;
        }
    }
    

    /**
     * @brief Returns a human readable c-string name of the `T` 
     */
    template <typename T>
    const char *TypeName()
    {
        static constexpr auto name = type_names::CexprTypeName<T>();
        return name.data();
    }


    template <typename _Container>
    void randomize(_Container &container, uint64_t seed = 0x27) {
        const number_t limit = 1.0 / sqrt(container.size()); 
		static int counter = 0;

        std::uniform_real_distribution<number_t> dist(-limit, limit);
		std::mt19937 eng(seed + counter++);

        // Radomize all values
        container.fill([&dist, &eng](){ return dist(eng); });
    }

    /**
     * @brief Initialize the `context` with random values (between +-1.0 / sqrt(input_size))
     * @param context must support `begin()` and `end()` functions
     */
    template <typename _ForwardIterator>
    void randomize(_ForwardIterator first, _ForwardIterator last, size_t input_size, uint64_t seed = 0x27)
    {
        const number_t limit = 1.0 / sqrt(input_size); 
		static int counter = 0;

        std::uniform_real_distribution<number_t> dist(-limit, limit);
		std::mt19937 eng(seed + counter++);

        // Radomize all values
        std::generate(first, last, [&dist, &eng](){ return dist(eng); });
    }


    /**
	* @brief Converts a number from a big-endian to a little-endian
    */
    inline int swap_32(int val)
    {
#if defined(__GNUC__) || defined(__clang__)
		return __builtin_bswap32(val);
#elif defined(_MSC_VER)
		return _byteswap_ulong(val);
#else
		return (val >> 24) | ((val << 8) & 0x00FF0000) | ((val >> 8) & 0x0000FF00) | (val << 24);
#endif
    }

    /**
     * @brief Get total size in bytes, with given alignment
     */
    template <size_t Alignment, typename dtype>
    constexpr size_t get_bytesize(size_t count) noexcept
    {
        return ((count * sizeof(dtype) + Alignment - 1) / Alignment ) * Alignment;
    }

    /**
     * @brief Allocate aligned memory with given count of elements (of dtype)
     * and aligment
     */
    template <size_t Alignment, typename dtype>
    dtype* aligned_alloc(size_t count) {
        return (dtype*) std::aligned_alloc(Alignment, get_bytesize<Alignment, dtype>(count));
    }

    template <typename dtype>
    void print_matrix(const dtype* A, size_t m, size_t n) {
        std::cout.setf(std::ios_base::fixed);
        auto prev = std::cout.precision();
        std::cout.precision(2);

        std::cout << "<matrix: dtype=" << utils::TypeName<dtype>() 
            << " dim=" << m << 'x' << n << ">\n";
        for (size_t i = 0; i < m; i++)
        {
            std::cout << "  [";
            for (size_t j = 0; j < n; j++)
            {
                std::cout << A[i * n + j];
                // print with proper formatting
                if (j != n - 1)
                    std::cout << ", ";
            }

            if (i != m - 1) {
                std::cout << "],\n";
            } else {
                std::cout << "]\n";
            }
        }
        std::cout.precision(prev);
    }   

    static int n_allocs = 0, max_allocs = 0, n_shared = 0, max_shared = 0;

    inline void inc_allocs(int i) {
        n_allocs += i;
        max_allocs = std::max(n_allocs, max_allocs);
    }

    inline void inc_shared(int i) {
        n_shared += i;
        max_shared = std::max(n_shared, max_shared);
    }
}
END_BLUST_NAMESPACE