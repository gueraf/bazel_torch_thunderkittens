load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")


def thunder_kittens_repository():
    git_repository(
        name="com_github_hazyresearch_thunderkittens",
        remote="https://github.com/HazyResearch/ThunderKittens.git",
        commit="6c27e28c8115d1839d9eeeb530913c184a75fc87",
        build_file="//third_party/thunder_kittens:BUILD.thunder_kittens.bzl",
        patch_cmds=[
            # Rename *.impl to *.inl
            "find . -name '*.impl' -exec sh -c 'mv \"$1\" \"${1%.impl}.inl\"' _ {} \\;",
            # Update #include statements from .impl to .inl
            "find . -type f \\( -name '*.h' -o -name '*.hpp' -o -name '*.cu' -o -name '*.cuh' -o -name '*.cc' -o -name '*.cpp' \\) -exec sed -i 's/#include \"\\([^\"]*\\)\\.impl\"/#include \"\\1.inl\"/g' {} \\;",
        ],
    )
