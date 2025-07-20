load("@rules_cuda//cuda:defs.bzl", "cuda_binary", "cuda_library", "cuda_test")

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

# ZLUDA vector addition library
cuda_library(
    name = "zluda_vector_add",
    srcs = ["zluda_vector_add.cu"],
    hdrs = ["zluda_vector_add.cuh"],
    visibility = ["//visibility:public"],
    deps = [
        "@zluda//:libcuda",
    ],
)

# ZLUDA vector addition test
cuda_test(
    name = "zluda_vector_add_test",
    size = "small",
    srcs = ["zluda_vector_add_test.cu"],
    linkopts = [
        "-Wl,-rpath,$$ORIGIN",
    ],
    deps = [
        ":zluda_vector_add",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "@zluda//:libcuda",
    ],
)

# ZLUDA vector addition demo binary
cuda_binary(
    name = "zluda_vector_add_demo",
    srcs = ["zluda_vector_add_demo.cpp"],
    linkopts = [
        "-Wl,-rpath,$$ORIGIN",
    ],
    deps = [
        ":zluda_vector_add",
        "@zluda//:libcuda",
    ],
)
