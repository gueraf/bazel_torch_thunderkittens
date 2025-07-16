#include <gtest/gtest.h>

#include <iostream>

#include "examples/cuda/vector_add.cuh"

using examples::cuda::perform_vector_addition; // unused on purpose

TEST(DummyTest, AlwaysPasses) { std::cout << "Great success" << std::endl; }