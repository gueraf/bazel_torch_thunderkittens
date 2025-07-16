# bazel build .vscode:all && ./bazel-bin/.vscode/vscode_import_generator
# https://github.com/microsoft/pylance-release/discussions/2712
import json
import os

RULES_PYTHON_PREFIXES = ["rules_python~~pip~pip_", "rules_python++pip+pip_"]
VSCODE_SETTINGS_FILE = ".vscode/settings.json"
PYREFLY_SETTINGS_FILE = "pyrefly.toml"

# search_path = ["bazel-template_py/external/rules_python~~pip~pip_313_absl_py/site-packages/"]


def main():
    # Example: /home/fabian/Desktop/manual_mlp/./bazel-bin/.vscode/vscode_import_generator.runfiles/.vscode_import_generator.venv/bin
    bazel_workspace = os.environ["PATH"].split(":")[0].split("/./bazel-bin/")[0].split("/")[-1]
    bazel_path = f"bazel-{bazel_workspace}/external"
    vscode_bazel_path = "${{workspaceFolder}}/{}".format(bazel_path)

    def check_word_start_with_right_prefix(word):
        return any(word.startswith(prefix) for prefix in RULES_PYTHON_PREFIXES)

    def add_prefix(word):
        return os.path.join(vscode_bazel_path, word)

    bazel_externals = os.listdir(bazel_path)
    filtered_list = [s for s in bazel_externals if check_word_start_with_right_prefix(s)]

    # Process VSCODE_SETTINGS_FILE
    vscode_list = [add_prefix(s) for s in filtered_list]
    vscode_list.sort()
    with open(VSCODE_SETTINGS_FILE, "r") as f:
        vscode_settings = json.loads(f.read())

    vscode_settings["python.analysis.extraPaths"] = vscode_list
    vscode_settings["python.autoComplete.extraPaths"] = vscode_list

    with open(VSCODE_SETTINGS_FILE, "w") as f:
        f.write(json.dumps(vscode_settings, indent=4, sort_keys=True))

    # Process PYREFLY_SETTINGS_FILE
    pyrefly_search_paths = ",".join(f'"{bazel_path}/{p}/site-packages/"' for p in filtered_list)
    if os.path.exists(PYREFLY_SETTINGS_FILE):
        with open(PYREFLY_SETTINGS_FILE, "r") as f:
            lines = f.readlines()

        new_lines = [
            (
                line
                if not line.startswith("search_path = [")
                else f"search_path = [{pyrefly_search_paths}]"
            )
            for line in lines
        ]

        with open(PYREFLY_SETTINGS_FILE, "w") as f:
            f.writelines(new_lines)


if __name__ == "__main__":
    main()
