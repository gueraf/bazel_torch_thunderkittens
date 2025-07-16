#!/bin/bash

# Check if bazelisk is available, otherwise use bazel
if command -v bazelisk &> /dev/null; then
    BAZEL_CMD="bazelisk"
    echo "Using bazelisk"
else
    BAZEL_CMD="bazel"
    echo "Using bazel"
fi

echo "Updating requirements_lock.txt" && \
$BAZEL_CMD run //:requirements && \
echo "Updating pypi_pkgs_lock.bzl" && \
$BAZEL_CMD query 'kind("alias rule", deps(@pip//...))' --output=label_kind | \
  grep ":pkg" | \
  grep -oP "@.+" | \
  sed -e "s/.*/'&',/" -e '1s/^/PIPY_PKGS = [/' -e '$s/,$/]/' > pypi_pkgs_lock.bzl && \
echo "Updating gazelle_python.yaml" && \
$BAZEL_CMD run //:gazelle_python_manifest.update && \
echo "Updating vscode settings.json" && \
$BAZEL_CMD build //.vscode:all && \
./bazel-bin/.vscode/vscode_import_generator
