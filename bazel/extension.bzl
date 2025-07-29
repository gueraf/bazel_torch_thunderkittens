"""CUDA architecture rules module extension."""

load("//bazel:defs.bzl", "cuda_arch_rules_repository")


def _cuda_arch_rules_extension_impl(ctx):
    """Implementation of the cuda_arch_rules module extension."""
    cuda_arch_rules_repository(name="cuda_arch_rules")


cuda_arch_rules_extension = module_extension(
    implementation=_cuda_arch_rules_extension_impl,
)
