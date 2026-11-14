#include "tensor.hpp"

#include <cstddef>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

class TestRunner {
public:
  void check(bool condition, std::string_view name) {
    ++checks;
    if (!condition) {
      ++failures;
      std::cerr << "FAIL: " << name << '\n';
    }
  }

  template <typename Expected, typename Action>
  void expect_exception(Action action, std::string_view name) {
    ++checks;
    try {
      action();
      ++failures;
      std::cerr << "FAIL: " << name << " (no exception)\n";
    } catch (const Expected &) {
      // Expected outcome.
    } catch (const std::exception &error) {
      ++failures;
      std::cerr << "FAIL: " << name << " (wrong exception: " << error.what()
                << ")\n";
    } catch (...) {
      ++failures;
      std::cerr << "FAIL: " << name << " (unknown exception)\n";
    }
  }

  int result() const {
    std::cout << "Tensor: " << (checks - failures) << '/' << checks
              << " checks passed\n";
    return failures == 0 ? 0 : 1;
  }

private:
  int checks{};
  int failures{};
};

int main() {
  TestRunner tests;

  try {
    Tensor<float> line({2});
    tests.check(line.get_shape() == std::vector<std::size_t>{2},
                "1D shape");
    tests.check(line.get_strides() == std::vector<std::size_t>{1},
                "1D stride");
    tests.check(line.get_size() == 2, "1D element count");
    tests.check(line[{0}] == 0.0f && line[{1}] == 0.0f,
                "1D elements start at zero");
    line[{1}] = -2.0f;
    tests.check(line[{0}] == 0.0f && line[{1}] == -2.0f,
                "1D writes affect only the selected element");

    Tensor<float> matrix({2, 3});
    tests.check(matrix.get_shape() == std::vector<std::size_t>({2, 3}),
                "2D shape");
    tests.check(matrix.get_strides() == std::vector<std::size_t>({3, 1}),
                "2D row-major strides");
    tests.check(matrix.get_size() == 6, "2D element count");
    matrix[{0, 2}] = 3.5f;
    matrix[{1, 0}] = 7.0f;
    tests.check(matrix[{0, 2}] == 3.5f && matrix[{1, 0}] == 7.0f &&
                    matrix[{0, 0}] == 0.0f,
                "2D coordinates address separate elements");

    Tensor<float> cube({2, 3, 4});
    tests.check(cube.get_shape() == std::vector<std::size_t>({2, 3, 4}),
                "3D shape");
    tests.check(cube.get_strides() == std::vector<std::size_t>({12, 4, 1}),
                "3D row-major strides");
    tests.check(cube.get_size() == 24, "3D element count");
    bool all_zero = true;
    for (std::size_t i = 0; i < 2; ++i) {
      for (std::size_t j = 0; j < 3; ++j) {
        for (std::size_t k = 0; k < 4; ++k) {
          all_zero = all_zero && cube[{i, j, k}] == 0.0f;
        }
      }
    }
    tests.check(all_zero, "all 3D elements start at zero");
    cube[{0, 2, 3}] = 1.5f;
    cube[{1, 0, 0}] = -4.0f;
    cube[{1, 2, 3}] = 8.0f;
    tests.check(cube[{0, 2, 3}] == 1.5f && cube[{1, 0, 0}] == -4.0f &&
                    cube[{1, 2, 3}] == 8.0f && cube[{0, 0, 0}] == 0.0f,
                "3D coordinates address separate elements");
    const Tensor<float> &read_only = cube;
    tests.check(read_only[{1, 2, 3}] == 8.0f,
                "const tensor supports reading");

    tests.expect_exception<std::invalid_argument>(
        [] { Tensor<float> invalid(std::vector<std::size_t>{}); },
        "empty shape is rejected");
    tests.expect_exception<std::invalid_argument>(
        [] { Tensor<float> invalid({2, 0, 4}); },
        "zero-sized dimension is rejected");
    tests.expect_exception<std::overflow_error>(
        [] { Tensor<float> invalid({std::numeric_limits<std::size_t>::max(), 2}); },
        "element count overflow is rejected");

    tests.expect_exception<std::out_of_range>(
        [&cube] { (void)cube[{0, 0}]; }, "too few indices are rejected");
    tests.expect_exception<std::out_of_range>(
        [&cube] { (void)cube[{0, 0, 0, 0}]; },
        "too many indices are rejected");
    tests.expect_exception<std::out_of_range>(
        [&cube] { (void)cube[{2, 0, 0}]; },
        "first dimension boundary is rejected");
    tests.expect_exception<std::out_of_range>(
        [&cube] { (void)cube[{0, 3, 0}]; },
        "middle dimension boundary is rejected");
    tests.expect_exception<std::out_of_range>(
        [&cube] { (void)cube[{0, 0, 4}]; },
        "last dimension boundary is rejected");
    tests.expect_exception<std::out_of_range>(
        [&read_only] { (void)read_only[{2, 0, 0}]; },
        "const tensor rejects out-of-range indices");

    Tensor<float> copied = cube;
    copied[{1, 2, 3}] = 9.0f;
    tests.check(cube[{1, 2, 3}] == 8.0f && copied[{1, 2, 3}] == 9.0f,
                "copy construction keeps values independent");
    Tensor<float> assigned({1});
    assigned = cube;
    assigned[{1, 2, 3}] = 10.0f;
    tests.check(assigned.get_shape() == cube.get_shape() &&
                    assigned.get_strides() == cube.get_strides() &&
                    assigned.get_size() == cube.get_size() &&
                    cube[{1, 2, 3}] == 8.0f && assigned[{1, 2, 3}] == 10.0f,
                "copy assignment copies shape and keeps values independent");
    Tensor<std::size_t> token_ids({2, 3});
    tests.check(token_ids.get_shape() == std::vector<std::size_t>({2, 3}) &&
                    token_ids.get_strides() == std::vector<std::size_t>({3, 1}) &&
                    token_ids.get_size() == 6,
                "integer tensor has row-major shape and strides");
    tests.check(token_ids[{0, 0}] == 0 && token_ids[{1, 2}] == 0,
                "integer tensor starts with zero values");
    token_ids[{1, 2}] = 255;
    tests.check(token_ids[{1, 2}] == 255 && token_ids[5] == 255 &&
                    token_ids[{0, 2}] == 0,
                "integer tensor supports coordinate and flat access");
    const Tensor<std::size_t> &read_only_ids = token_ids;
    tests.check(read_only_ids[{1, 2}] == 255,
                "const integer tensor supports reading");
    Tensor<std::size_t> copied_ids = token_ids;
    copied_ids[{1, 2}] = 7;
    tests.check(token_ids[{1, 2}] == 255 && copied_ids[{1, 2}] == 7,
                "integer tensor copies keep values independent");
    tests.expect_exception<std::out_of_range>(
        [&token_ids] { (void)token_ids[{2, 0}]; },
        "integer tensor rejects out-of-range coordinates");

  } catch (const std::exception &error) {
    std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
    return 1;
  } catch (...) {
    std::cerr << "FAIL: unexpected unknown exception\n";
    return 1;
  }

  return tests.result();
}
