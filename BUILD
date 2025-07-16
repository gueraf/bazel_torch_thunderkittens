load("@gazelle//:def.bzl", "gazelle")
load("@pip//:requirements.bzl", "all_whl_requirements")
load("@rules_python_gazelle_plugin//manifest:defs.bzl", "gazelle_python_manifest")
load("@rules_python_gazelle_plugin//modules_mapping:def.bzl", "modules_mapping")
load("@rules_uv//uv:pip.bzl", "pip_compile")

pip_compile(
    name = "requirements",
    requirements_in = "//:requirements.txt",
    requirements_txt = "//:requirements_lock.txt",
)

exports_files(["requirements_lock.txt"])

modules_mapping(
    name = "modules_map",
    wheels = all_whl_requirements,
)

# bazel run //:gazelle_python_manifest.update
gazelle_python_manifest(
    name = "gazelle_python_manifest",
    modules_mapping = ":modules_map",
    pip_repository_name = "pip",
    requirements = "//:requirements_lock.txt",
)

# bazel run //:gazelle
gazelle(
    name = "gazelle",
    gazelle = "@rules_python_gazelle_plugin//python:gazelle_binary",
)

# gazelle:python_root
# gazelle:python_binary_naming_convention $package_name$
# gazelle:python_generation_mode file
# gazelle:python_default_visibility NONE
# gazelle:map_kind py_binary py_binary @aspect_rules_py//py:defs.bzl
# gazelle:map_kind py_library py_library @aspect_rules_py//py:defs.bzl
# gazelle:map_kind py_test py_test @aspect_rules_py//py:defs.bzl
