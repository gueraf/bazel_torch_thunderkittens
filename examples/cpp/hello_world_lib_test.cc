#include "examples/cpp/hello_world_lib.h"

#include <gtest/gtest.h>

#include <iostream>

namespace examples::cpp {
namespace {

TEST(HelloWorldLibTest, HelloWorld) {
  EXPECT_EQ(get_hello_world(), "Hello, World!");
}

}  // namespace
}  // namespace examples::cpp
