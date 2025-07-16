// See
// https://github.com/HazyResearch/ThunderKittens/tree/main/kernels/matmul/H100

#include <chrono>
#include <iostream>
#include <vector>

#include "examples/thunder_kittens/tk_gmm_utils.h"

using examples::thunder_kittens::make_matrix;

int main() {
  constexpr int K = 128;
  constexpr int L = 256;
  constexpr int M = 512;
  constexpr float alpha = 2.0f;
  constexpr float beta = 3.0f;

  std::vector<float> A = make_matrix(K, L);
  std::vector<float> B = make_matrix(L, M);
  std::vector<float> C(K * M, 1.0f);  // Initialize C with ones

  auto start = std::chrono::high_resolution_clock::now();
  examples::thunder_kittens::gmm(A, B, alpha, beta, K, L, M, C);
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Execution time (CPU): "
            << std::chrono::duration<double, std::milli>(end - start).count()
            << " ms" << std::endl;

  return 0;
}
