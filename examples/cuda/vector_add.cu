#include "examples/cuda/vector_add.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

namespace examples::cuda
{

    // CUDA kernel for vector addition
    __global__ void vector_add(const float *a, const float *b, float *c, int n)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
        {
            c[idx] = a[idx] + b[idx];
        }
    }

    void perform_vector_addition(int N)
    {
        const size_t size = N * sizeof(float);

        // Host vectors
        float *h_a = (float *)malloc(size);
        float *h_b = (float *)malloc(size);
        float *h_c = (float *)malloc(size);
        float *h_result = (float *)malloc(size); // For verification

        if (!h_a || !h_b || !h_c || !h_result)
        {
            fprintf(stderr, "Failed to allocate host memory\n");
            exit(1);
        }

        // Initialize host vectors
        for (int i = 0; i < N; i++)
        {
            h_a[i] = (float)i;
            h_b[i] = (float)(i * 2);
            h_result[i] = h_a[i] + h_b[i]; // Expected result
        }

        // Device vectors
        float *d_a, *d_b, *d_c;

        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_a, size));
        CUDA_CHECK(cudaMalloc(&d_b, size));
        CUDA_CHECK(cudaMalloc(&d_c, size));

        // Copy host vectors to device
        CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

        // Launch kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

        printf("Launching kernel with %d blocks and %d threads per block\n",
               blocksPerGrid, threadsPerBlock);

        // Create CUDA events for timing
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // Record start time
        CUDA_CHECK(cudaEventRecord(start));

        vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

        // Record stop time
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        // Calculate elapsed time
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        // Check for kernel launch errors
        CUDA_CHECK(cudaGetLastError());

        // Copy result from device to host
        CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

        // Verify the result
        bool success = true;
        const float epsilon = 1e-5f;
        for (int i = 0; i < N; i++)
        {
            if (fabs(h_c[i] - h_result[i]) > epsilon)
            {
                fprintf(stderr, "Verification failed at index %d: got %f, expected %f\n",
                        i, h_c[i], h_result[i]);
                success = false;
                break;
            }
        }

        if (success)
        {
            printf("Vector addition completed successfully!\n");
            printf("Processed %d elements in %.3f ms\n", N, milliseconds);
            printf("Bandwidth: %.2f GB/s\n", (3.0f * size) / (milliseconds * 1e6f));
        }
        else
        {
            printf("Vector addition failed verification!\n");
        }

        // Cleanup
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_c));

        free(h_a);
        free(h_b);
        free(h_c);
        free(h_result);
    }

} // namespace examples::cuda