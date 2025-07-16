#ifndef VECTOR_ADD_H
#define VECTOR_ADD_H

#include <cuda_runtime.h>

namespace examples::cuda
{

// Helper macro to check CUDA errors
#define CUDA_CHECK(call)                                                      \
    do                                                                        \
    {                                                                         \
        cudaError_t error = call;                                             \
        if (error != cudaSuccess)                                             \
        {                                                                     \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error));                               \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

    // Function declarations
    __global__ void vector_add(const float *a, const float *b, float *c, int n);

    // Host function to perform vector addition with timing and verification
    void perform_vector_addition(int N);

} // namespace examples::cuda

#endif // VECTOR_ADD_H