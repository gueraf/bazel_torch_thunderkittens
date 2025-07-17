#!/bin/bash

# CircleCI Config Generator Script
# This script generates a dynamic CircleCI configuration based on Bazel test targets

set -e

echo "Generating CircleCI configuration..."

# Get all Python test targets
echo "Querying Python test targets..."
PY_TEST_TARGETS=$(bazelisk query 'kind("py_test", //...)')

echo "Querying C++ test targets..."
# Get C++ test targets
CC_TEST_TARGETS=$(bazelisk query 'kind("cc_test", //...)')

# Combine all test targets and filter out requirements_test
echo "Combining and filtering test targets..."
TEST_TARGETS=$(echo "$PY_TEST_TARGETS $CC_TEST_TARGETS" | tr ' ' '\n' | grep -vE ":requirements_test")

echo "Found test targets:"
echo "$TEST_TARGETS"
echo ""

# Start generating the new config
echo "Generating CircleCI configuration file..."
echo "version: 2.1" > .circleci/generated_config.yml
echo "" >> .circleci/generated_config.yml
echo "jobs:" >> .circleci/generated_config.yml

# Add a job for each test target
for target in $TEST_TARGETS; do
  # Create a sanitized job name from the target
  job_name=$(echo "$target" | tr '/:' '__' | tr -d '()' | sed 's/^_//g')
  
  # Ensure job name starts with a letter (CircleCI requirement)
  if [[ ! $job_name =~ ^[A-Za-z] ]]; then
    job_name="job-${job_name}"
  fi
  
  echo "  ${job_name}:" >> .circleci/generated_config.yml
  echo "    docker:" >> .circleci/generated_config.yml
  echo "      - image: gueraf/cuda_bazel:latest" >> .circleci/generated_config.yml
  echo "        auth:" >> .circleci/generated_config.yml
  echo "          username: \$DOCKER_USERNAME" >> .circleci/generated_config.yml
  echo "          password: \$DOCKER_ACCESS_TOKEN" >> .circleci/generated_config.yml
  echo "    steps:" >> .circleci/generated_config.yml
  echo "      - checkout" >> .circleci/generated_config.yml
  echo "      - run:" >> .circleci/generated_config.yml
  echo "          name: \"Run Bazel test: ${target}\"" >> .circleci/generated_config.yml
  echo "          command: \"bazelisk test --build_tests_only ${target}\"" >> .circleci/generated_config.yml
done

# Add workflow configuration
echo "" >> .circleci/generated_config.yml
echo "workflows:" >> .circleci/generated_config.yml
echo "  test-workflow:" >> .circleci/generated_config.yml
echo "    jobs:" >> .circleci/generated_config.yml

# Add all jobs to workflow
for target in $TEST_TARGETS; do
  job_name=$(echo "$target" | tr '/:' '__' | tr -d '()' | sed 's/^_//g')
  
  # Ensure job name starts with a letter (CircleCI requirement)
  if [[ ! $job_name =~ ^[A-Za-z] ]]; then
    job_name="job-${job_name}"
  fi
  
  echo "      - ${job_name}" >> .circleci/generated_config.yml
done

echo ""
echo "Configuration generated successfully: generated_config.yml"
echo ""
echo "Generated configuration:"
echo "========================"
cat .circleci/generated_config.yml
