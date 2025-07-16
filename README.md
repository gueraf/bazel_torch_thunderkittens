# template_py

A template to start hobby projects (Python, Bazel, uv, VSCode, Pyrefly).

[![CircleCI](https://dl.circleci.com/status-badge/img/circleci/MoqoviNBD61M3rSAe4Bj3m/AuCfEk66ESdUMz8A135Wdf/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/circleci/MoqoviNBD61M3rSAe4Bj3m/AuCfEk66ESdUMz8A135Wdf/tree/main)

# Getting Started
## Install Bazel(isk)
```bash
sudo apt install npm && \
sudo npm install -g @bazel/bazelisk
```

## Add and Update Dependencies
```bash
echo "example-pypi-package>=0.1.0" >> requirements.txt && \
./update_requirements.sh && \
git add requirements* .vscode/settings.json gazelle_python.yaml pypi_pkgs_lock.bzl
```

## Build, Test, and Run
```bash
bazel run //examples/py_hello_world:hello_world && \
bazel test //examples/py_hello_world:hello_world_test
```

## Update BUILD files
```bash
bazel run //:gazelle
```
