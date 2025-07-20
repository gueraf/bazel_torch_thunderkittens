load("@rules_cuda//cuda:defs.bzl", "cuda_binary", "cuda_library", "cuda_test")

# ZLUDA example binary - configurable between CUDA and ZLUDA
cuda_binary(
    name = "zluda_example",
    srcs = ["zluda_example.cpp"],
    linkopts = [
        "-Wl,-rpath,$$ORIGIN",  # Allow finding shared libraries at runtime
        "-ldl",  # Link against dl library for dlopen functions
    ],
    deps = [
        "//cuda_config:cuda_runtime",  # Configurable CUDA/ZLUDA dependency
    ],
)

# System CUDA version of the example (always uses system CUDA)
cuda_binary(
    name = "cuda_example",
    srcs = ["zluda_example.cpp"],
    linkopts = [
        "-Wl,-rpath,$$ORIGIN",
        "-ldl",
    ],
    deps = [
        "@rules_cuda//cuda:runtime",  # Always use system CUDA
    ],
)

# ZLUDA-specific version (always uses ZLUDA)
cuda_binary(
    name = "zluda_only_example",
    srcs = ["zluda_example.cpp"],
    linkopts = [
        "-Wl,-rpath,$$ORIGIN",
        "-ldl",
    ],
    deps = [
        "@zluda//:libcuda",  # Always use ZLUDA
    ],
)

# ZLUDA vector addition library - configurable
cuda_library(
    name = "zluda_vector_add",
    srcs = ["zluda_vector_add.cu"],
    hdrs = ["zluda_vector_add.cuh"],
    visibility = ["//visibility:public"],
    deps = [
        "//cuda_config:cuda_runtime",  # Configurable CUDA/ZLUDA dependency
    ],
)

# System CUDA version of vector addition
cuda_library(
    name = "cuda_vector_add",
    srcs = ["zluda_vector_add.cu"],
    hdrs = ["zluda_vector_add.cuh"],
    visibility = ["//visibility:public"],
    deps = [
        "@rules_cuda//cuda:runtime",  # Always use system CUDA
    ],
)

# ZLUDA-only version of vector addition
cuda_library(
    name = "zluda_only_vector_add",
    srcs = ["zluda_vector_add.cu"],
    hdrs = ["zluda_vector_add.cuh"],
    visibility = ["//visibility:public"],
    deps = [
        "@zluda//:libcuda",  # Always use ZLUDA
    ],
)

# ZLUDA vector addition test
# TODO: Enable once working.
# cuda_test(
#     name = "zluda_vector_add_test",
#     size = "small",
#     srcs = ["zluda_vector_add_test.cu"],
#     linkopts = [
#         "-Wl,-rpath,$$ORIGIN",
#     ],
#     deps = [
#         ":zluda_vector_add",
#         "@googletest//:gtest",
#         "@googletest//:gtest_main",
#         "@zluda//:libcuda",
#     ],
# )

# ZLUDA vector addition demo binary - configurable
cuda_binary(
    name = "zluda_vector_add_demo",
    srcs = ["zluda_vector_add_demo.cpp"],
    linkopts = [
        "-Wl,-rpath,$$ORIGIN",
    ],
    deps = [
        ":zluda_vector_add",
        "//cuda_config:cuda_runtime",
    ],
)

# System CUDA version of the demo
cuda_binary(
    name = "cuda_vector_add_demo",
    srcs = ["zluda_vector_add_demo.cpp"],
    linkopts = [
        "-Wl,-rpath,$$ORIGIN",
    ],
    deps = [
        ":cuda_vector_add",
        "@rules_cuda//cuda:runtime",
    ],
)

# ZLUDA-only version of the demo
cuda_binary(
    name = "zluda_only_vector_add_demo",
    srcs = ["zluda_vector_add_demo.cpp"],
    linkopts = [
        "-Wl,-rpath,$$ORIGIN",
    ],
    deps = [
        ":zluda_only_vector_add",
        "@zluda//:libcuda",
    ],
)
