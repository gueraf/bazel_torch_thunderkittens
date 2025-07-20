load("@rules_cuda//cuda:defs.bzl", "cuda_binary")

# ZLUDA example binary that links against ZLUDA's libcuda.so
# This demonstrates how to use ZLUDA as a drop-in replacement for CUDA
cuda_binary(
    name = "zluda_example",
    srcs = ["zluda_example.cpp"],
    linkopts = [
        "-Wl,-rpath,$$ORIGIN",  # Allow finding shared libraries at runtime
    ],
    deps = [
        "@zluda//:libcuda",  # Links against zluda/libcuda.so.1
    ],
)
