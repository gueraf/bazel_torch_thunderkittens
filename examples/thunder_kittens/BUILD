load("@rules_cuda//cuda:defs.bzl", "cuda_library", "cuda_test")

cuda_test(
    name = "build_test",
    srcs = ["build_test.cu"],
    deps = [
        ":tk_gmm",
        "//third_party/thunder_kittens:kittens",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)

cc_binary(
    name = "tk_gmm_benchmark",
    srcs = ["tk_gmm_benchmark.cc"],
    copts = [
        "-O3",
        "-fopenmp",
    ],
    deps = [":tk_gmm_utils"],
)

cc_library(
    name = "tk_gmm_utils",
    srcs = ["tk_gmm_utils.cc"],
    hdrs = ["tk_gmm_utils.h"],
    deps = [
        "@abseil-cpp//absl/algorithm:container",
    ],
)

cc_test(
    name = "tk_gmm_utils_test",
    size = "small",
    srcs = ["tk_gmm_utils_test.cc"],
    deps = [
        ":tk_gmm_utils",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)

cuda_library(
    name = "tk_gmm",
    srcs = ["tk_gmm.cu"],
    hdrs = ["tk_gmm.cuh"],
    deps = ["//third_party/thunder_kittens:kittens"],
)
