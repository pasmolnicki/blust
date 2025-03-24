#include <gtest/gtest.h>
#include <blust/blust.hpp>

using namespace blust;

class BackendTest : public ::testing::Test
{
protected:
	cuda_backend cuda;
	cpu_backend cpu;

	void SetUp() override {
		cuda.init(0, nullptr);
	}
};


TEST_F(BackendTest, TestCudaVecAdd)
{
	matrix_t mat1({ 8, 16 }, .0f);
	matrix_t mat2({ 8, 16 }, .0f);
	matrix_t res1({ 8, 16 }, .0f);
	matrix_t res2({ 8, 16 }, .0f);

	utils::randomize(mat1.begin(), mat1.end(), mat1.size());
	utils::randomize(mat2.begin(), mat2.end(), mat2.size());

	cuda.vector_add(res1.data(), mat1.data(), mat2.data(), mat1.size());
	cpu.vector_add(res2.data(), mat1.data(), mat2.data(), mat1.size());

	ASSERT_EQ(res1, res2);
}

TEST_F(BackendTest, TestCudaVecSub)
{
	matrix_t mat1({ 8, 16 }, .0f);
	matrix_t mat2({ 8, 16 }, .0f);
	matrix_t res1({ 8, 16 }, .0f);
	matrix_t res2({ 8, 16 }, .0f);

	utils::randomize(mat1.begin(), mat1.end(), mat1.size());
	utils::randomize(mat2.begin(), mat2.end(), mat2.size());

	cuda.vector_sub(res1.data(), mat1.data(), mat2.data(), mat1.size());
	cpu.vector_sub(res2.data(), mat1.data(), mat2.data(), mat1.size());

	ASSERT_EQ(res1, res2);
}

TEST_F(BackendTest, TestCudaVecMulHadamard)
{
	matrix_t mat1({ 8, 16 }, .0f);
	matrix_t mat2({ 8, 16 }, .0f);
	matrix_t res1({ 8, 16 }, .0f);
	matrix_t res2({ 8, 16 }, .0f);

	utils::randomize(mat1.begin(), mat1.end(), mat1.size());
	utils::randomize(mat2.begin(), mat2.end(), mat2.size());

	cuda.vector_mul_hadamard(res1.data(), mat1.data(), mat2.data(), mat1.size());
	cpu.vector_mul_hadamard(res2.data(), mat1.data(), mat2.data(), mat1.size());
	ASSERT_EQ(res1, res2);
}

TEST_F(BackendTest, TestCudaVecScalarMul)
{
	matrix_t mat1({ 8, 16 }, .0f);
	matrix_t res1({ 8, 16 }, .0f);
	matrix_t res2({ 8, 16 }, .0f);

	utils::randomize(mat1.begin(), mat1.end(), mat1.size());

	number_t scalar = 0.5f;

	cuda.vector_scalar_mul(res1.data(), mat1.data(), scalar, mat1.size());
	cpu.vector_scalar_mul(res2.data(), mat1.data(), scalar, mat1.size());

	ASSERT_EQ(res1, res2);
}

TEST_F(BackendTest, TestCudaMatTranspose)
{
	matrix_t mat1({ 8, 16 }, .0f);
	matrix_t res1({ 16, 8 }, .0f);
	matrix_t res2({ 16, 8 }, .0f);

	utils::randomize(mat1.begin(), mat1.end(), mat1.size());

	cuda.mat_transpose(res1.data(), mat1.data(), 8, 16);
	cpu.mat_transpose(res2.data(), mat1.data(), 8, 16);

	ASSERT_EQ(res1, res2);
}

TEST_F(BackendTest, TestCudaMatMul)
{
	matrix_t mat1({ 8, 16 }, .0f);
	matrix_t mat2({ 16, 8 }, .0f);
	matrix_t res1({ 8, 8 }, .0f);
	matrix_t res2({ 8, 8 }, .0f);

	utils::randomize(mat1.begin(), mat1.end(), mat1.size());
	utils::randomize(mat2.begin(), mat2.end(), mat2.size());

	cuda.mat_mul(res1.data(), mat1.data(), mat2.data(), 8, 8, 16);
	cpu.mat_mul(res2.data(), mat1.data(), mat2.data(), 8, 8, 16);

	ASSERT_EQ(res1, res2);
}