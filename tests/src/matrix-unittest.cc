#include <gtest/gtest.h>


#include <blust/blust.hpp>
using namespace blust;

class MatrixFixture : public testing::Test
{
protected:
    static constexpr int n_matrices = 1024;
    static constexpr int m_size     = 512;
    std::vector<matrix<int>> m;
    std::vector<std::vector<int>> v;

    void SetUp()
    {
        std::uniform_int_distribution<size_t> value_gen(-512, 512);
        std::mt19937 rd(0x144258);

        m.reserve(n_matrices);
        v.reserve(n_matrices);

        for (size_t i = 0; i < n_matrices; i++)
        {
            m.push_back(matrix<int>({m_size, m_size}));
            v.push_back(std::vector<int>(m_size));
    
            const size_t size = m[i].size();
            for (size_t j = 0; j < size; j++)
                m[i](j) = value_gen(rd);
            
            for(size_t j = 0; j < v[i].size(); j++)
                v[i][j] = value_gen(rd);
        }
    }
};


TEST_F(MatrixFixture, TestSpeedVectorMultiplication)
{
    for (size_t i = 0; i < n_matrices; i++)
    {
        auto r = m[i] * v[i];
    }
}

TEST(Matrix, TestMultiplicationByConst)
{
    matrix<int> m({
        { 1, 2}, 
        {-1, 3}, 
        { 3, 1}
    });

    int k = 2;

    ASSERT_EQ(m * k, matrix<int>({
        { 2, 4},
        {-2, 6},
        { 6, 2}
    }));
}


TEST(Matrix, TestMultiplyMatrices)
{
    matrix<int> m({
        { 1, 2}, 
        {-1, 3}, 
        { 3, 1}
    });

    matrix<int> d({
        { 5, 3,  5, 2,  1},
        {-1, 5, 10, 8, -7}
    });

    ASSERT_EQ(m * d, matrix<int>({
        { 3, 13, 25, 18, -13},
        {-8, 12, 25, 22, -22},
        {14, 14, 25, 14, -4}
    }));
}

TEST(Matrix, TestMultiplyMatrixByVector)
{
    matrix<int> m({
        { 5, 3,  5, 2,  1},
        {-1, 5, 10, 8, -7}
    });

    std::vector<int> v({2, 7, -5, 9, 1});

    auto res = m * v;
    auto expect = std::vector<int>({25, 48});
    ASSERT_EQ(res, expect);
}

TEST(Matrix, TestHadamard)
{
    matrix<int> m1({
        {1, 2, 3},
        {1, 2, 3}
    });

    matrix<int> m2({
        {3, 2, -1},
        {3, -2, 1},
    });

    auto res = m1 % m2;
    auto expect = matrix<int>{
        {3, 4, -3},
        {3, -4, 3},
    };

    ASSERT_EQ(res, expect);
}
