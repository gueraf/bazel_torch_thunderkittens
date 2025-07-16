# load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_cuda//cuda:defs.bzl", "cuda_library")

package(default_visibility=["//visibility:public"])

cuda_library(
    name="kittens",
    hdrs=glob(
        [
            "include/**/*.cuh",
            "include/**/*.inl",
            "prototype/**/*.cuh",
        ]
    ),
    includes=["include", "prototype"],
    copts=[
        # "-std=c++20",
        "-DKITTENS_HOPPER",
        # "-Xcompiler",
        # "-Wno-error=maybe-uninitialized",
        # "-Wno-deprecated-gpu-targets",
        # "-Xcompiler=-fPIE",
    ],
    host_copts=[
        # "-std=c++20",
        "-DKITTENS_HOPPER",
        # "-Xcompiler",
        # "-Wno-error=maybe-uninitialized",
        # "-Xcompiler=-fPIE",
    ],
    visibility=["//visibility:public"],
)
