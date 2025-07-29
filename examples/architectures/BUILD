load("//bazel:cuda_arch_rules.bzl", "cuda_library_4090", "cuda_library_h100", "cuda_test_4090", "cuda_test_h100")

cuda_library_h100(
    name = "simple_kernel_h100",
    srcs = ["simple_kernel.cu"],
    hdrs = ["simple_kernel.cuh"],
    deps = [],
)

cuda_test_h100(
    name = "simple_kernel_test_h100",
    srcs = ["simple_kernel_test.cu"],
    deps = [
        ":simple_kernel_h100",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)

cuda_library_4090(
    name = "simple_kernel_4090",
    srcs = ["simple_kernel.cu"],
    hdrs = ["simple_kernel.cuh"],
    deps = [],
)

cuda_test_4090(
    name = "simple_kernel_test_4090",
    srcs = ["simple_kernel_test.cu"],
    deps = [
        ":simple_kernel_4090",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
