load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

cc_library(
    name = "hello_world_lib",
    srcs = ["hello_world_lib.cc"],
    hdrs = ["hello_world_lib.h"],
)

cc_test(
    name = "hello_world_lib_test",
    size = "small",
    srcs = ["hello_world_lib_test.cc"],
    deps = [
        ":hello_world_lib",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
