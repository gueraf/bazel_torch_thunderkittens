#!/bin/bash

# Build and push script for AMD CUDA Docker image
set -e

IMAGE_NAME="gueraf/rocm_cuda:latest"

echo "Building Docker image: ${IMAGE_NAME}"

# Build the image
docker build -f .circleci/Dockerfile_amd_cuda -t "${IMAGE_NAME}" .

echo "Image built successfully: ${IMAGE_NAME}"

# Ask for confirmation before pushing
read -p "Do you want to push the image to the registry? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Pushing image to registry..."
    docker push "${IMAGE_NAME}"
    echo "Image pushed successfully: ${IMAGE_NAME}"
else
    echo "Skipping push. Image is available locally as: ${IMAGE_NAME}"
fi

echo "Done!"
