#include <gtest/gtest.h>
#include <blust/tensor.hpp>

TEST(TensorTest, DefaultConstructor) 
{
    blust::tensor t;

    EXPECT_TRUE(t.empty());
    EXPECT_EQ(t.size(), 0u);
    EXPECT_FALSE(t.is_cuda());
}

TEST(TensorTest, ConstructWithShape) 
{
    blust::shape s({2, 3});
    blust::tensor t(s);

    EXPECT_FALSE(t.empty());
    EXPECT_EQ(t.size(), 6u);
    EXPECT_FALSE(t.is_cuda());
}

TEST(TensorTest, InitializationValue) 
{
    blust::shape s({3});
    blust::tensor t(s, 4.0);
    auto data = t.data();

    for (size_t i = 0; i < t.size(); ++i) {
        EXPECT_EQ(data[i], 4.0);
    }
}

TEST(TensorTest, CopyConstructor) 
{
    blust::shape s({2});
    blust::tensor t(s, 2.0);
    blust::tensor copy(t);

    EXPECT_EQ(copy.size(), t.size());
    EXPECT_FALSE(copy.empty());

    auto data = copy.data();
    for (size_t i = 0; i < copy.size(); ++i) {
        EXPECT_EQ(data[i], 2.0);
    }
}

TEST(TensorTest, MoveConstructor) 
{
    blust::shape s({2});
    blust::tensor t(s, 3.0);
    blust::tensor moved(std::move(t));

    EXPECT_EQ(moved.size(), 2u);
    EXPECT_FALSE(moved.empty());
}

TEST(TensorTest, ReleaseMethod) 
{
    blust::shape s({4});
    blust::tensor t(s, 7.0);

    auto ptr = t.release();
    EXPECT_TRUE(t.empty());
    EXPECT_EQ(ptr[0], 7.0);
    std::free(ptr);
}