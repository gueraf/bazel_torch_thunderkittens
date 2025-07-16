// See
// https://github.com/HazyResearch/ThunderKittens/tree/main/kernels/matmul/H100

#include <iostream>

#include "examples/thunder_kittens/tk_gmm_utils.h"

using examples::thunder_kittens::make_matrix;

int main() {
  constexpr int K = 1024;
  constexpr int L = 2048;
  constexpr int M = 4096;

  std::vector<float> A = make_matrix(K, L);
  std::vector<float> B = make_matrix(L, M);

  std::cout << "Great success" << std::endl;
  return 0;
}
