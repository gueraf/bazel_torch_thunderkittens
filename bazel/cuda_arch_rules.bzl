load(
    "@rules_cuda//cuda:defs.bzl",
    _cuda_library="cuda_library",
    _cuda_binary="cuda_binary",
    _cuda_test="cuda_test",
)


# H100
def cuda_library_h100(name, copts=None, **kwargs):
    if copts == None:
        copts = []

    final_copts = copts + ["-DKITTENS_HOPPER"]

    # Override the CUDA architecture configuration
    kwargs["exec_properties"] = kwargs.get("exec_properties", {})
    kwargs["exec_properties"]["@rules_cuda//cuda:archs"] = "compute_90a"

    _cuda_library(name=name, copts=final_copts, **kwargs)


def cuda_binary_h100(name, copts=None, **kwargs):
    if copts == None:
        copts = []

    final_copts = copts + ["-DKITTENS_HOPPER"]

    # Override the CUDA architecture configuration
    kwargs["exec_properties"] = kwargs.get("exec_properties", {})
    kwargs["exec_properties"]["@rules_cuda//cuda:archs"] = "compute_90a"

    _cuda_binary(name=name, copts=final_copts, **kwargs)


def cuda_test_h100(name, copts=None, **kwargs):
    if copts == None:
        copts = []

    final_copts = copts + ["-DKITTENS_HOPPER"]

    # Override the CUDA architecture configuration
    kwargs["exec_properties"] = kwargs.get("exec_properties", {})
    kwargs["exec_properties"]["@rules_cuda//cuda:archs"] = "compute_90a"

    _cuda_test(name=name, copts=final_copts, **kwargs)


# 4090
def cuda_library_4090(name, copts=None, **kwargs):
    if copts == None:
        copts = []

    final_copts = copts + ["-DKITTENS_4090"]

    # Override the CUDA architecture configuration
    kwargs["exec_properties"] = kwargs.get("exec_properties", {})
    kwargs["exec_properties"]["@rules_cuda//cuda:archs"] = "compute_89"

    _cuda_library(name=name, copts=final_copts, **kwargs)


def cuda_binary_4090(name, copts=None, **kwargs):
    if copts == None:
        copts = []

    final_copts = copts + ["-DKITTENS_4090"]

    # Override the CUDA architecture configuration
    kwargs["exec_properties"] = kwargs.get("exec_properties", {})
    kwargs["exec_properties"]["@rules_cuda//cuda:archs"] = "compute_89"

    _cuda_binary(name=name, copts=final_copts, **kwargs)


def cuda_test_4090(name, copts=None, **kwargs):
    if copts == None:
        copts = []

    final_copts = copts + ["-DKITTENS_4090"]

    # Override the CUDA architecture configuration
    kwargs["exec_properties"] = kwargs.get("exec_properties", {})
    kwargs["exec_properties"]["@rules_cuda//cuda:archs"] = "compute_89"

    _cuda_test(name=name, copts=final_copts, **kwargs)
