#include <gtest/gtest.h>

#include <blust/blust.hpp>


int main(int argc, char** argv)
{
	blust::init(argc, argv);
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}