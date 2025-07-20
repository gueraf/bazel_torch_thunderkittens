#!/bin/bash

# Test ZLUDA integration in a ROCm-enabled Docker container
echo "Testing ZLUDA integration with configurable backends..."

docker run gueraf/rocm_cuda:latest \
    bash -c "ROCM_PATH=\$(ls -d /opt/rocm* | head -1) && \
             export LD_LIBRARY_PATH=\$ROCM_PATH/lib:\$LD_LIBRARY_PATH && \
             export PATH=\$ROCM_PATH/bin:\$PATH && \
             echo \"Using ROCm installation at: \$ROCM_PATH\" && \
             echo \"Available AMD COMGR libraries:\" && \
             ls -la \$ROCM_PATH/lib/libamd_comgr* 2>/dev/null || echo \"No libamd_comgr libraries found\" && \
             echo \"Creating symlinks for missing library versions...\" && \
             cd \$ROCM_PATH/lib && \
             if [ -f libamd_comgr.so.3 ] && [ ! -f libamd_comgr.so.2 ]; then \
                 ln -sf libamd_comgr.so.3 libamd_comgr.so.2; \
                 echo \"Created symlink: libamd_comgr.so.2 -> libamd_comgr.so.3\"; \
             elif [ -f libamd_comgr.so ] && [ ! -f libamd_comgr.so.2 ]; then \
                 ln -sf libamd_comgr.so libamd_comgr.so.2; \
                 echo \"Created symlink: libamd_comgr.so.2 -> libamd_comgr.so\"; \
             fi && \
             cd /tmp && \
             git clone https://github.com/gueraf/bazel_torch_thunderkittens.git && \
             cd bazel_torch_thunderkittens && \
             echo '=== Testing ZLUDA-only target ===' && \
             bazelisk run //examples/zluda:zluda_only_example && \
             echo -e '\n=== Building ZLUDA vector addition demo ===' && \
             bazelisk build //examples/zluda:zluda_only_vector_add_demo"

# TODO: bazelisk run //examples/zluda:zluda_example --define cuda_backend=zluda actually loads CUDA backend.