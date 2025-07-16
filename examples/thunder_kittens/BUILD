load("@rules_cuda//cuda:defs.bzl", "cuda_test")

cuda_test(
    name = "build_test",
    srcs = ["build_test.cu"],
    # copts = [
    #     "-std=c++20",
    #     "-Xcompiler",
    #     # "-Wno-error=maybe-uninitialized",
    #     # "-Xcompiler=-fPIE",
    # ],
    # host_copts = [
    #     "-std=c++20",
    #     "-Xcompiler",
    #     # "-Wno-error=maybe-uninitialized",
    #     # "-Xcompiler=-fPIE",
    # ],
    deps = [
        "//third_party/thunder_kittens:kittens",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)

cc_binary(
    name = "tk_gmm_benchmark",
    srcs = ["tk_gmm_benchmark.cc"],
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
