load("@aspect_rules_py//py:defs.bzl", "py_binary", "py_library", "py_test")

py_binary(
    name = "hello_world",
    srcs = ["hello_world.py"],
    deps = [
        ":hello_world_lib",
        "@pip//absl_py",
    ],
)

py_library(
    name = "hello_world_lib",
    srcs = ["hello_world_lib.py"],
)

py_test(
    name = "hello_world_test",
    srcs = ["hello_world_test.py"],
    deps = [
        ":hello_world_lib",
        "@pip//absl_py",
    ],
)
