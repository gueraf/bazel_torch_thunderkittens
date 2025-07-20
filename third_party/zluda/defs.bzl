load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")


def zluda_repository():
    """Downloads and configures ZLUDA."""
    http_archive(
        name="zluda",
        url="https://github.com/vosen/ZLUDA/releases/download/v4/zluda-4-linux.tar.gz",
        sha256="72894bed3ef94263b24eb63898ae711b8f98adda5518443a9289b9cf506ce03b",
        strip_prefix="zluda",
        build_file="//third_party/zluda:BUILD.zluda",
    )
