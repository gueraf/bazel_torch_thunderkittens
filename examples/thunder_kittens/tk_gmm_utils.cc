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

}  // namespace examples::thunder_kittens