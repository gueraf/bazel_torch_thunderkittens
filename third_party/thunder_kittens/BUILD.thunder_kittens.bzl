# load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_cuda//cuda:defs.bzl", "cuda_library")
load(
    "@cuda_arch_rules//:cuda_arch_rules.bzl",
    "cuda_library_h100",
    "cuda_library_4090",
)

package(default_visibility=["//visibility:public"])

cuda_library_h100(
    name="kittens_h100",
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
        # "-DKITTENS_HOPPER",
        # "-Xcompiler",
        # "-Wno-error=maybe-uninitialized",
        # "-Wno-deprecated-gpu-targets",
        # "-Xcompiler=-fPIE",
    ],
    host_copts=[
        # "-std=c++20",
        # "-DKITTENS_HOPPER",
        # "-Xcompiler",
        # "-Wno-error=maybe-uninitialized",
        # "-Xcompiler=-fPIE",
    ],
    visibility=["//visibility:public"],
)

cuda_library_4090(
    name="kittens_4090",
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
        # "-DKITTENS_HOPPER",
        # "-Xcompiler",
        # "-Wno-error=maybe-uninitialized",
        # "-Wno-deprecated-gpu-targets",
        # "-Xcompiler=-fPIE",
    ],
    host_copts=[
        # "-std=c++20",
        # "-DKITTENS_HOPPER",
        # "-Xcompiler",
        # "-Wno-error=maybe-uninitialized",
        # "-Xcompiler=-fPIE",
    ],
    visibility=["//visibility:public"],
)
