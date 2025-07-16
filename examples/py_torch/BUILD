load("@aspect_rules_py//py:defs.bzl", "py_binary", "py_test")

py_binary(
    name = "load_pytorch",
    srcs = ["load_pytorch.py"],
    deps = [
        "@pip//absl_py",
        "@pip//numpy",
        "@pip//torch",
    ],
)

py_test(
    name = "pytorch_test",
    srcs = ["pytorch_test.py"],
    deps = [
        "@pip//absl_py",
        "@pip//torch",
    ],
)
