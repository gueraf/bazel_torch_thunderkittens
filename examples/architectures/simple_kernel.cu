#include "examples/architectures/simple_kernel.cuh"

#include <iostream>
#include <string>

namespace examples::architectures {
namespace {

__global__ void increment_all_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] += 1.0;
    }
}

}  // namespace

void increment_all(float* data, int size) {
    #ifdef KITTENS_HOPPER
    constexpr std::string_view kArch = "hopper";
    #endif

    #ifdef KITTENS_4090
    constexpr std::string_view kArch = "4090";
    #endif

    std::cout << "Running increment_all on " << kArch << " architecture." << std::endl;

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    increment_all_kernel<<<numBlocks, blockSize>>>(data, size);
    cudaDeviceSynchronize(); // Ensure kernel execution is complete
}

}  // namespace examples::architectures
