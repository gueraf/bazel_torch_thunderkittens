#!/bin/bash

# Test ZLUDA integration in a ROCm-enabled Docker container
echo "Testing ZLUDA integration with configurable backends..."

docker run gueraf/rocm_cuda:latest \
    bash -c "cd /tmp && \
             git clone https://github.com/gueraf/bazel_torch_thunderkittens.git && \
             cd bazel_torch_thunderkittens && \
             echo '=== Testing ZLUDA-only target ===' && \
             bazelisk run //examples/zluda:zluda_only_example && \
             echo -e '\n=== Testing configurable target with ZLUDA ===' && \
             bazelisk run //examples/zluda:zluda_example --define cuda_backend=zluda && \
             echo -e '\n=== Testing ZLUDA vector addition demo ===' && \
             bazelisk run //examples/zluda:zluda_only_vector_add_demo"
