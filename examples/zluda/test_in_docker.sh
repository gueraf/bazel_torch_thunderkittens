#!/bin/bash

# Test ZLUDA integration in a ROCm-enabled Docker container
echo "Testing ZLUDA integration with configurable backends..."

docker run rocm/pytorch:rocm6.4.1_ubuntu24.04_py3.12_pytorch_release_2.7.1 \
    bash -c "apt update && apt install -y npm && npm install -g @bazel/bazelisk && \
             cd /tmp && \
             git clone https://github.com/gueraf/bazel_torch_thunderkittens.git && \
             cd bazel_torch_thunderkittens && \
             echo '=== Testing ZLUDA-only target ===' && \
             bazelisk run //examples/zluda:zluda_only_example && \
             echo -e '\n=== Testing configurable target with ZLUDA ===' && \
             bazelisk run //examples/zluda:zluda_example --define cuda_backend=zluda && \
             echo -e '\n=== Testing ZLUDA vector addition demo ===' && \
             bazelisk run //examples/zluda:zluda_only_vector_add_demo"
