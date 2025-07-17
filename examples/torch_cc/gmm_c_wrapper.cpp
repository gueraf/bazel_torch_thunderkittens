#include <vector>

#include "examples/thunder_kittens/tk_gmm_utils.h"

extern "C" {

// C wrapper function that can be called from ctypes
void gmm_c_wrapper(const float* A, const float* B, float alpha, float beta,
                   int K, int L, int M, float* C) {
  // Convert arrays to vectors
  std::vector<float> A_vec(A, A + K * L);
  std::vector<float> B_vec(B, B + L * M);
  std::vector<float> C_vec(C, C + K * M);

  // Call the original GMM function
  examples::thunder_kittens::gmm(A_vec, B_vec, alpha, beta, K, L, M, C_vec);

  // Copy result back to output array
  std::copy(C_vec.begin(), C_vec.end(), C);
}

}  // extern "C"
