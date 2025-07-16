// See
// https://github.com/HazyResearch/ThunderKittens/tree/main/kernels/matmul/H100

#include <chrono>
#include <iostream>
#include <vector>

#include "examples/thunder_kittens/tk_gmm.cuh"
#include "examples/thunder_kittens/tk_gmm_utils.h"

using ::examples::thunder_kittens::gmm;
using ::examples::thunder_kittens::gmm_gpu;
using ::examples::thunder_kittens::make_matrix;

int main() {
  constexpr int K = 128;
  constexpr int L = 256;
  constexpr int M = 512;
  constexpr float alpha = 1.0f;
  constexpr float beta = 1.0f;

  std::vector<float> A = make_matrix(K, L);
  std::vector<float> B = make_matrix(L, M);
  std::vector<float> C(K * M, 0.0f);

  // CPU
  auto start = std::chrono::high_resolution_clock::now();
  gmm(A, B, alpha, beta, K, L, M, C);
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Execution time (CPU): "
            << std::chrono::duration<double, std::milli>(end - start).count()
            << " ms" << std::endl;

  // GPU
  // TODO: Alpha and beta not supported.
  gmm_gpu(A.data(), B.data(), alpha, beta, K, L, M, C.data());

  return 0;
}
