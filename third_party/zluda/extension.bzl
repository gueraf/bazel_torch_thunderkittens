"""ZLUDA module extension."""

load("//third_party/zluda:defs.bzl", "zluda_repository")


def _zluda_extension_impl(ctx):
    """Implementation of the ZLUDA module extension."""
    zluda_repository()


zluda_extension = module_extension(
    implementation=_zluda_extension_impl,
)
