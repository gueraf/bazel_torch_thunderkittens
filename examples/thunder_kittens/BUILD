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
