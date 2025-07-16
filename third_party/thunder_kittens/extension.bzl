"""ThunderKittens module extension."""

load("//third_party/thunder_kittens:defs.bzl", "thunder_kittens_repository")


def _thunder_kittens_extension_impl(ctx):
    """Implementation of the thunder_kittens module extension."""
    thunder_kittens_repository()


thunder_kittens_extension = module_extension(
    implementation=_thunder_kittens_extension_impl,
)
