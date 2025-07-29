load("@cuda_arch_rules//:cuda_arch_rules.bzl", "cuda_binary_h100", "cuda_library_h100", "cuda_test_4090", "cuda_test_h100")
load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

cuda_test_h100(
    name = "build_test_h100",
    srcs = ["build_test.cu"],
    deps = [
        "//third_party/thunder_kittens:kittens_h100",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)

cuda_test_4090(
    name = "build_test_4090",
    srcs = ["build_test.cu"],
    deps = [
        "//third_party/thunder_kittens:kittens_4090",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)

cuda_binary_h100(
    name = "tk_gmm_benchmark",
    srcs = ["tk_gmm_benchmark.cc"],
    copts = [
        "-DNDEBUG",
        # "-fPIE",
        # "-fopenmp",
        # "-Wno-psabi",
        # "-fno-strict-aliasing",
        "-std=c++20",
        # "-DKITTENS_HOPPER",
        "--expt-extended-lambda",
        "--expt-relaxed-constexpr",
        "--use_fast_math",
    ],
    linkopts = [
        "-lcuda",
        # "-lcudadevrt",
        # "-lcudart_static",
        # "-lcublas",
        "-lgomp",
        "-lrt",
        "-lpthread",
        "-ldl",
    ],
    deps = [
        ":tk_gmm",
        ":tk_gmm_utils",
    ],
)

cc_library(
    name = "tk_gmm_utils",
    srcs = ["tk_gmm_utils.cc"],
    hdrs = ["tk_gmm_utils.h"],
    copts = ["-fPIC"],  # Ensure position-independent code
    visibility = [
        "//examples/torch_cc:__subpackages__",
    ],
    deps = [
        "@abseil-cpp//absl/algorithm:container",
    ],
    # alwayslink = True,  # Force linking of all symbols
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

cuda_library_h100(
    name = "tk_gmm",
    srcs = ["tk_gmm.cu"],
    hdrs = ["tk_gmm.cuh"],
    deps = ["//third_party/thunder_kittens:kittens_h100"],
)

cuda_test_h100(
    name = "tk_gmm_test",
    srcs = ["tk_gmm_test.cu"],
    deps = [
        ":tk_gmm",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
