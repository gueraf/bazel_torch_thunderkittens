load("@cuda_arch_rules//:cuda_arch_rules.bzl", "cuda_binary_4090", "cuda_binary_h100", "cuda_library_4090", "cuda_library_h100")

cuda_library_h100(
    name = "attention_h100",
    hdrs = ["attention_h100.cuh"],
    deps = [
        "//third_party/thunder_kittens:kittens_h100",
    ],
)

cuda_binary_h100(
    name = "attention_h100_main",
    srcs = ["attention_h100_main.cu"],
    deps = [
        ":attention_h100",
    ],
)

cuda_library_4090(
    name = "attention_4090",
    hdrs = ["attention_4090.cuh"],
    deps = [
        "//third_party/thunder_kittens:kittens_4090",
    ],
)

cuda_binary_4090(
    name = "attention_4090_main",
    srcs = ["attention_4090_main.cu"],
    deps = [
        ":attention_4090",
    ],
)
