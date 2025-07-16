#include "examples/thunder_kittens/tk_gmm_utils.h"

#include <gtest/gtest.h>

namespace examples::thunder_kittens {
namespace {

TEST(TkGmmUtilsTest, MakeMatric) {
  auto matrix = make_matrix(3, 4);
  ASSERT_EQ(matrix.size(), 12);
  for (const auto& value : matrix) {
    ASSERT_GE(value, -0.5f);
    ASSERT_LE(value, 0.5f);
  }
}

}  // namespace
}  // namespace examples::thunder_kittens
