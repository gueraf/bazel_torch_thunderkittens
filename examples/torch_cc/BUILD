load("@aspect_rules_py//py:defs.bzl", "py_library", "py_test")
load("@rules_cc//cc:defs.bzl", "cc_library")

# C wrapper for the GMM function
cc_library(
    name = "gmm_c_wrapper",
    srcs = ["gmm_c_wrapper.cpp"],
    copts = [
        "-std=c++17",
        "-fPIC",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//examples/thunder_kittens:tk_gmm_utils",
    ],
)

# Python library wrapping the GMM function
py_library(
    name = "gmm_torch",
    srcs = ["gmm_torch.py"],
    data = [":gmm_c_wrapper"],
    visibility = ["//visibility:public"],
    deps = [
        "@pip//numpy",
        "@pip//torch",
    ],
)

# Python test for the wrapper
py_test(
    name = "gmm_torch_test",
    srcs = ["gmm_torch_test.py"],
    deps = [
        ":gmm_torch",
        "@pip//absl_py",
        "@pip//numpy",
        "@pip//torch",
    ],
)

# Demo showing how to use the GMM wrapper
py_binary(
    name = "gmm_demo",
    srcs = ["gmm_demo.py"],
    deps = [
        ":gmm_torch",
        "@pip//numpy",
        "@pip//torch",
    ],
)
