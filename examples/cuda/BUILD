load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_test")
load("@rules_cuda//cuda:defs.bzl", "cuda_library", "cuda_test")

cuda_library(
    name = "vector_add_lib",
    srcs = ["vector_add.cu"],
    hdrs = ["vector_add.cuh"],
    visibility = ["//visibility:public"],
)

cuda_test(
    name = "vector_add_test",
    size = "small",
    srcs = ["vector_add_test.cu"],
    deps = [
        ":vector_add_lib",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
