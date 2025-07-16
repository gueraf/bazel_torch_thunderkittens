#include <vector>

namespace examples::thunder_kittens {

float* to_device(const std::vector<float>& vec);

void gmm_gpu(const float* A, const float* B, float alpha, float beta, int M,
             int K, int N, float* C);

}  // namespace examples::thunder_kittens