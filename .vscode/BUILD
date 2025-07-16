# gazelle:ignore

load("@aspect_rules_py//py:defs.bzl", "py_binary")
load("//:pypi_pkgs_lock.bzl", "PIPY_PKGS")

py_binary(
    name = "fake_binary",
    srcs = ["fake_binary.py"],
    deps = PIPY_PKGS,
)

py_binary(
    name = "vscode_import_generator",
    srcs = ["vscode_import_generator.py"],
)
