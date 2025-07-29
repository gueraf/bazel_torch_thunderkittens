#include <gtest/gtest.h>

#include <iostream>
#include "examples/architectures/simple_kernel.cuh"

namespace examples::architectures {
    TEST(SimpleKernelTest, Increment) { 
        std::vector<float> data(1000, 0.0f);

        float* d_data;
        cudaMalloc(&d_data, data.size() * sizeof(float));
        cudaMemcpy(d_data, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);

        increment_all(d_data, data.size());

        cudaMemcpy(data.data(), d_data, data.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_data);
        
        // for (const auto& value : data) {
        //     EXPECT_EQ(value, 1.0f);
        // }
    }
}  // namespace examples::architectures
