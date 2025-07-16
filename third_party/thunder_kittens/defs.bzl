load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")


def thunder_kittens_repository():
    git_repository(
        name="com_github_hazyresearch_thunderkittens",
        remote="https://github.com/HazyResearch/ThunderKittens.git",
        commit="aaab847f430ed313ed466e64b25b9177babd1db8",
        build_file_content="""
# load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_cuda//cuda:defs.bzl", "cuda_library")

package(default_visibility = ["//visibility:public"])

cuda_library(
    name = "kittens",
    hdrs = glob(["include/**/*.cuh", "prototype/**/*.cuh"]),
    includes = ["include", "prototype"],
    # copts = [
    #     "-std=c++20",
    #     "-Xcompiler",
    #     "-Wno-error=maybe-uninitialized",
    #     "-Wno-deprecated-gpu-targets",
    #     "-Xcompiler=-fPIE",
    # ],
    # host_copts = [
    #     "-std=c++20",
    #     "-Xcompiler",
    #     "-Wno-error=maybe-uninitialized",
    #     "-Xcompiler=-fPIE",
    # ],
    visibility = ["//visibility:public"],
)
""",
    )
