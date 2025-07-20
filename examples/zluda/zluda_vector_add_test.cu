#include <gtest/gtest.h>

#include <iostream>

#include "examples/zluda/zluda_vector_add.cuh"

using namespace examples::zluda;

class ZludaVectorAddTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize CUDA context
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {
      GTEST_SKIP() << "CUDA/ZLUDA not available: " << cudaGetErrorString(error);
    }

    if (deviceCount == 0) {
      GTEST_SKIP() << "No CUDA/ZLUDA devices found";
    }

    std::cout << "Found " << deviceCount << " CUDA/ZLUDA device(s)"
              << std::endl;
  }
};

TEST_F(ZludaVectorAddTest, SmallVectorAddition) {
  EXPECT_TRUE(perform_vector_addition(100));
}

TEST_F(ZludaVectorAddTest, MediumVectorAddition) {
  EXPECT_TRUE(perform_vector_addition(1000));
}

TEST_F(ZludaVectorAddTest, LargeVectorAddition) {
  EXPECT_TRUE(perform_vector_addition(10000));
}

TEST_F(ZludaVectorAddTest, VeryLargeVectorAddition) {
  EXPECT_TRUE(perform_vector_addition(1000000));
}

// Test the verify_result function directly
TEST(ZludaVectorAddUnitTest, VerifyResultCorrect) {
  const int n = 5;
  float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  float b[] = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  float c[] = {3.0f, 5.0f, 7.0f, 9.0f, 11.0f};  // correct results

  EXPECT_TRUE(verify_result(a, b, c, n));
}

TEST(ZludaVectorAddUnitTest, VerifyResultIncorrect) {
  const int n = 5;
  float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  float b[] = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  float c[] = {3.0f, 5.0f, 7.0f, 9.0f, 12.0f};  // last element is wrong

  EXPECT_FALSE(verify_result(a, b, c, n));
}
