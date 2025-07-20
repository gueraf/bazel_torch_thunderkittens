#include <iostream>

#include "examples/zluda/zluda_vector_add.cuh"

int main() {
  std::cout << "ZLUDA Vector Addition Demo" << std::endl;
  std::cout << "=========================" << std::endl;

  // Test different vector sizes
  int test_sizes[] = {100, 1000, 10000, 100000};
  int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);

  bool all_passed = true;

  for (int i = 0; i < num_tests; i++) {
    std::cout << "\nTest " << (i + 1) << "/" << num_tests << ":" << std::endl;
    bool result = examples::zluda::perform_vector_addition(test_sizes[i]);

    if (!result) {
      all_passed = false;
      std::cout << "❌ Test failed for size " << test_sizes[i] << std::endl;
    } else {
      std::cout << "✅ Test passed for size " << test_sizes[i] << std::endl;
    }
  }

  std::cout << "\n=========================" << std::endl;
  if (all_passed) {
    std::cout
        << "🎉 All tests passed! ZLUDA vector addition is working correctly."
        << std::endl;
    return 0;
  } else {
    std::cout << "❌ Some tests failed." << std::endl;
    return 1;
  }
}
