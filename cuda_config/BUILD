# Configuration for selecting between CUDA and ZLUDA
# Simple configuration without requiring bazel_skylib

# Config setting for ZLUDA (use --define cuda_backend=zluda)
config_setting(
    name = "use_zluda",
    define_values = {"cuda_backend": "zluda"},
)

# Config setting for system CUDA (use --define cuda_backend=system_cuda or default)
config_setting(
    name = "use_system_cuda",
    define_values = {"cuda_backend": "system_cuda"},
)

# Selectable CUDA library dependency
alias(
    name = "cuda_runtime",
    actual = select({
        ":use_zluda": "@zluda//:libcuda",
        "//conditions:default": "@rules_cuda//cuda:runtime",
    }),
    visibility = ["//visibility:public"],
)

# Selectable CUDA driver dependency
alias(
    name = "cuda_driver",
    actual = select({
        ":use_zluda": "@zluda//:libcuda",
        "//conditions:default": "@rules_cuda//cuda:runtime",
    }),
    visibility = ["//visibility:public"],
)
