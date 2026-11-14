#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
cd "$project_root"
source "$script_dir/lib.sh"

build_dir="$script_dir/artifacts"
mkdir -p "$build_dir"

compiler="${CXX:-g++}"
flags=(-std=c++23 -Wall -Wextra -Werror -Wpedantic -Wshadow -Wconversion -ICore/Inc -g)
test_files=(tests/cpp_tests/*.cpp)
core_sources=()

for source in Core/Src/*.cpp; do
    case "$source" in
        Core/Src/main.cpp) ;;
        *) core_sources+=("$source") ;;
    esac
done

if (( ${#test_files[@]} == 0 )); then
    printf 'No C++ tests found in tests/cpp_tests.\n' >&2
    exit 1
fi

for test_file in "${test_files[@]}"; do
    test_name="${test_file##*/}"
    test_binary="$build_dir/${test_name%.cpp}"

    if ! "$compiler" "${flags[@]}" "${core_sources[@]}" "$test_file" -o "$test_binary"; then
        printf 'BUILD FAILED: %s\n' "$test_file" >&2
        exit 1
    fi

    assert_passes "$test_binary" || true
done

print_result
