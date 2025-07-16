#include <vector>

namespace examples::thunder_kittens {

std::vector<float> make_matrix(int rows, int cols);

// C = alpha AB + beta C
// A: K x L, B: L x M, C: K x M
// alpha, beta: scalars
void gmm(const std::vector<float>& A, const std::vector<float>& B, float alpha,
         float beta, int K, int L, int M, std::vector<float>& C);

}  // namespace examples::thunder_kittens