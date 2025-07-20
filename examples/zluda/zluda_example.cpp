// Simple example demonstrating how to use ZLUDA
#include <cuda_runtime.h>

#include <iostream>

int main() {
  std::cout << "ZLUDA Example - Testing CUDA Runtime" << std::endl;

  // Get CUDA device count
  int deviceCount = 0;
  cudaError_t error = cudaGetDeviceCount(&deviceCount);

  if (error != cudaSuccess) {
    std::cout << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
    return 1;
  }

  std::cout << "Number of CUDA devices: " << deviceCount << std::endl;

  if (deviceCount > 0) {
    cudaDeviceProp prop;
    for (int i = 0; i < deviceCount; i++) {
      cudaGetDeviceProperties(&prop, i);
      std::cout << "Device " << i << ": " << prop.name << std::endl;
    }
  }

  return 0;
}
