#!/usr/bin/env bash

CPP_VER="-std=c++20"
COMPILE_FLAGS="-Wall -Wextra -Werror -Wpedantic -Wshadow -Wconversion"
BUILD_DIR="build"
SRC_DIR="Core/Src"

g++ $CPP_VER $COMPILE_FLAGS -ICore/Inc -g $SRC_DIR/*.cpp -o $BUILD_DIR/llm