#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>

#include "examples/zluda/zluda_vector_add.cuh"

namespace examples::zluda {

// CUDA kernel for vector addition
__global__ void vector_add_kernel(const float *a, const float *b, float *c,
                                  int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

bool verify_result(const float *a, const float *b, const float *c, int n,
                   float tolerance) {
  for (int i = 0; i < n; i++) {
    float expected = a[i] + b[i];
    if (fabs(c[i] - expected) > tolerance) {
      printf("Verification failed at index %d: expected %f, got %f\n", i,
             expected, c[i]);
      return false;
    }
  }
  return true;
}

bool perform_vector_addition(int N) {
  std::cout << "ZLUDA Vector Addition Example (N=" << N << ")" << std::endl;

  const size_t size = N * sizeof(float);

  // Host vectors
  float *h_a = (float *)malloc(size);
  float *h_b = (float *)malloc(size);
  float *h_c = (float *)malloc(size);

  if (!h_a || !h_b || !h_c) {
    fprintf(stderr, "Failed to allocate host memory\n");
    return false;
  }

  // Initialize host vectors
  for (int i = 0; i < N; i++) {
    h_a[i] = (float)i;
    h_b[i] = (float)(i * 2);
  }

  // Device vectors
  float *d_a, *d_b, *d_c;

  // Allocate device memory
  CUDA_CHECK(cudaMalloc(&d_a, size));
  CUDA_CHECK(cudaMalloc(&d_b, size));
  CUDA_CHECK(cudaMalloc(&d_c, size));

  // Copy data from host to device
  CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

  // Launch kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  std::cout << "Launching kernel with " << blocksPerGrid << " blocks, "
            << threadsPerBlock << " threads per block" << std::endl;

  vector_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

  // Check for kernel launch error
  CUDA_CHECK(cudaGetLastError());

  // Wait for kernel to complete
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy result back to host
  CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

  // Verify result
  bool success = verify_result(h_a, h_b, h_c, N);

  if (success) {
    std::cout << "Vector addition completed successfully!" << std::endl;
  } else {
    std::cout << "Vector addition verification failed!" << std::endl;
  }

  // Print first few results for verification
  std::cout << "First 5 results:" << std::endl;
  for (int i = 0; i < std::min(5, N); i++) {
    std::cout << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << std::endl;
  }

  // Cleanup
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  free(h_a);
  free(h_b);
  free(h_c);

  return success;
}

}  // namespace examples::zluda
