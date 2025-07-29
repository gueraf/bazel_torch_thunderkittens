"""Repository rule for CUDA architecture rules."""


def _cuda_arch_rules_repository_impl(repository_ctx):
    """Implementation of the cuda_arch_rules repository rule."""
    # Create a symlink to the local cuda_arch_rules.bzl file
    repository_ctx.symlink(
        repository_ctx.path(repository_ctx.attr._cuda_arch_rules_bzl), "cuda_arch_rules.bzl"
    )
    # Create an empty BUILD file
    repository_ctx.file("BUILD", "")


cuda_arch_rules_repository = repository_rule(
    implementation=_cuda_arch_rules_repository_impl,
    attrs={
        "_cuda_arch_rules_bzl": attr.label(
            default="//bazel:cuda_arch_rules.bzl",
            allow_single_file=True,
        ),
    },
)
