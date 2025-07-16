FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

ENV CUDA_HOME=/usr/local/cuda/

# Update package list and install npm
RUN apt-get update && \
    apt-get install -y npm curl jq && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Bazelisk globally via npm
RUN npm install -g @bazel/bazelisk
