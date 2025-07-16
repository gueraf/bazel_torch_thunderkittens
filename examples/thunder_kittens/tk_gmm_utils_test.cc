#include "examples/thunder_kittens/tk_gmm_utils.h"

#include <gtest/gtest.h>

namespace examples::thunder_kittens {
namespace {

TEST(TkGmmUtilsTest, MakeMatrix) {
  auto matrix = make_matrix(3, 4);
  ASSERT_EQ(matrix.size(), 12);
  for (const auto& value : matrix) {
    ASSERT_GE(value, -0.5f);
    ASSERT_LE(value, 0.5f);
  }
}

TEST(TkGmmUtilsTest, GMM) {
  constexpr int K = 2, L = 3, M = 4;
  constexpr float alpha = 2.0f, beta = 3.0f;
  std::vector<float> A = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};  // 2x3 matrix
  std::vector<float> B = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                          7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};  // 3x4 matrix
  std::vector<float> C(K * M, 1.0f);  // Initialize C with ones

  gmm(A, B, alpha, beta, K, L, M, C);

  // Expected result matrix C (2x4):
  // C = alpha * (A * B) + beta * original_C
  // where original_C was initialized with ones

  // Calculate expected values:
  // A * B results in:
  // Row 0: [1*1+2*5+3*9=38, 1*2+2*6+3*10=44, 1*3+2*7+3*11=50, 1*4+2*8+3*12=56]
  // Row 1: [4*1+5*5+6*9=83, 4*2+5*6+6*10=98, 4*3+5*7+6*11=113,
  // 4*4+5*8+6*12=128]

  // With alpha=2.0 and beta=3.0 (original C values were 1.0):
  // C[0,0] = alpha * 38 + beta * 1 = 2*38 + 3*1 = 76 + 3 = 79
  ASSERT_FLOAT_EQ(C[0], 79.0f);
  // C[0,1] = alpha * 44 + beta * 1 = 2*44 + 3*1 = 88 + 3 = 91
  ASSERT_FLOAT_EQ(C[1], 91.0f);
  // C[0,2] = alpha * 50 + beta * 1 = 2*50 + 3*1 = 100 + 3 = 103
  ASSERT_FLOAT_EQ(C[2], 103.0f);
  // C[0,3] = alpha * 56 + beta * 1 = 2*56 + 3*1 = 112 + 3 = 115
  ASSERT_FLOAT_EQ(C[3], 115.0f);
  // C[1,0] = alpha * 83 + beta * 1 = 2*83 + 3*1 = 166 + 3 = 169
  ASSERT_FLOAT_EQ(C[4], 169.0f);
  // C[1,1] = alpha * 98 + beta * 1 = 2*98 + 3*1 = 196 + 3 = 199
  ASSERT_FLOAT_EQ(C[5], 199.0f);
  // C[1,2] = alpha * 113 + beta * 1 = 2*113 + 3*1 = 226 + 3 = 229
  ASSERT_FLOAT_EQ(C[6], 229.0f);
  // C[1,3] = alpha * 128 + beta * 1 = 2*128 + 3*1 = 256 + 3 = 259
  ASSERT_FLOAT_EQ(C[7], 259.0f);
}

}  // namespace
}  // namespace examples::thunder_kittens
