#include "examples/thunder_kittens/tk_gmm_utils.h"

#include <random>

#include "absl/algorithm/container.h"

namespace examples::thunder_kittens {

std::vector<float> make_matrix(int rows, int cols) {
  // Initialize random number generator
  std::random_device rd;
  std::mt19937 gen(42);
  std::uniform_real_distribution<> dis(-0.5, 0.5);

  std::vector<float> matrix(rows * cols);
  absl::c_generate(matrix, [&]() { return dis(gen); });
  return matrix;
}

void gmm(const std::vector<float>& A, const std::vector<float>& B, float alpha,
         float beta, int K, int L, int M, std::vector<float>& C) {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < M; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < L; ++k) {
        sum += A[i * L + k] * B[k * M + j];
      }
      C[i * M + j] = alpha * sum + beta * C[i * M + j];
    }
  }
}

}  // namespace examples::thunder_kittens