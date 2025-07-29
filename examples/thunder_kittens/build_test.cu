#include <gtest/gtest.h>

#include <iostream>

#include "kittens.cuh"

// TODO: Investigate why this fails on 4090
#ifndef KITTENS_4090
#include "prototype.cuh"
#endif

TEST(DummyTest, AlwaysPasses) { std::cout << "Great success" << std::endl; }
