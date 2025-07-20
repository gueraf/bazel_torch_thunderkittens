#ifndef ZLUDA_VECTOR_ADD_H
#define ZLUDA_VECTOR_ADD_H

#include <cuda_runtime.h>

namespace examples::zluda {

// Helper macro to check CUDA errors
#define CUDA_CHECK(call)                                                \
  do {                                                                  \
    cudaError_t error = call;                                           \
    if (error != cudaSuccess) {                                         \
      fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(error));                               \
      exit(1);                                                          \
    }                                                                   \
  } while (0)

// Function declarations
__global__ void vector_add_kernel(const float *a, const float *b, float *c,
                                  int n);

// Host function to perform vector addition with ZLUDA
bool perform_vector_addition(int N);

// Function to verify vector addition result
bool verify_result(const float *a, const float *b, const float *c, int n,
                   float tolerance = 1e-5f);

}  // namespace examples::zluda

#endif  // ZLUDA_VECTOR_ADD_H
