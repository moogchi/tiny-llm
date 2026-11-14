#include "math.hpp"

#include <cstddef>
#include <iostream>
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

  template <typename Action>
  void expect_invalid_argument(Action action, std::string_view name) {
    ++checks;
    try {
      action();
      ++failures;
      std::cerr << "FAIL: " << name << " (no exception)\n";
    } catch (const std::invalid_argument &) {
      // Expected outcome.
    } catch (const std::exception &error) {
      ++failures;
      std::cerr << "FAIL: " << name << " (wrong exception: " << error.what()
                << ")\n";
    }
  }

  int result() const {
    std::cout << "Linear: " << (checks - failures) << '/' << checks
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
    Tensor<float> weights({2, 3});
    weights[{0, 0}] = 1.0f;
    weights[{0, 1}] = 4.0f;
    weights[{0, 2}] = -1.0f;
    weights[{1, 0}] = -1.0f;
    weights[{1, 1}] = 2.0f;
    weights[{1, 2}] = 3.0f;
    Tensor<float> bias({3});
    bias[0] = 5.0f;
    bias[1] = -2.0f;
    bias[2] = 1.0f;

    Tensor<float> one_dimensional({2});
    one_dimensional[0] = 2.0f;
    one_dimensional[1] = 3.0f;
    const Tensor<float> one_dimensional_result =
        forward_pass(one_dimensional, weights, bias);
    tests.check(one_dimensional_result.get_shape() ==
                    std::vector<std::size_t>{3},
                "1D input produces a 1D output");
    tests.check(one_dimensional_result[0] == 4.0f &&
                    one_dimensional_result[1] == 12.0f &&
                    one_dimensional_result[2] == 8.0f,
                "1D result matches a hand-calculated reference");

    Tensor<float> two_dimensional({2, 2});
    two_dimensional[{0, 0}] = 2.0f;
    two_dimensional[{0, 1}] = 3.0f;
    two_dimensional[{1, 0}] = 1.0f;
    two_dimensional[{1, 1}] = 0.0f;
    const Tensor<float> two_dimensional_result =
        forward_pass(two_dimensional, weights, bias);
    tests.check(two_dimensional_result.get_shape() ==
                    std::vector<std::size_t>({2, 3}),
                "2D input replaces only its last dimension");
    tests.check(two_dimensional_result[{0, 0}] == 4.0f &&
                    two_dimensional_result[{0, 1}] == 12.0f &&
                    two_dimensional_result[{0, 2}] == 8.0f,
                "first 2D row matches the reference");
    tests.check(two_dimensional_result[{1, 0}] == 6.0f &&
                    two_dimensional_result[{1, 1}] == 2.0f &&
                    two_dimensional_result[{1, 2}] == 0.0f,
                "bias is added to the second 2D row");

    Tensor<float> three_dimensional({2, 2, 2});
    three_dimensional[{0, 0, 0}] = 2.0f;
    three_dimensional[{0, 0, 1}] = 3.0f;
    three_dimensional[{0, 1, 0}] = 1.0f;
    three_dimensional[{0, 1, 1}] = 0.0f;
    three_dimensional[{1, 0, 0}] = 0.0f;
    three_dimensional[{1, 0, 1}] = 0.0f;
    three_dimensional[{1, 1, 0}] = -1.0f;
    three_dimensional[{1, 1, 1}] = 4.0f;
    const Tensor<float> three_dimensional_result =
        forward_pass(three_dimensional, weights, bias);
    tests.check(three_dimensional_result.get_shape() ==
                    std::vector<std::size_t>({2, 2, 3}),
                "3D input replaces only its last dimension");
    tests.check(three_dimensional_result[{0, 0, 0}] == 4.0f &&
                    three_dimensional_result[{0, 1, 1}] == 2.0f &&
                    three_dimensional_result[{1, 0, 0}] == 5.0f &&
                    three_dimensional_result[{1, 0, 1}] == -2.0f &&
                    three_dimensional_result[{1, 0, 2}] == 1.0f &&
                    three_dimensional_result[{1, 1, 0}] == 0.0f &&
                    three_dimensional_result[{1, 1, 1}] == 2.0f &&
                    three_dimensional_result[{1, 1, 2}] == 14.0f,
                "3D rows use the same weights and bias independently");
    tests.check(three_dimensional[{1, 1, 0}] == -1.0f &&
                    weights[{1, 2}] == 3.0f && bias[0] == 5.0f,
                "linear forward leaves input, weights, and bias unchanged");

    Tensor<float> four_dimensional({2, 1, 2, 2});
    four_dimensional[0] = 2.0f;
    four_dimensional[1] = 3.0f;
    four_dimensional[2] = 1.0f;
    four_dimensional[3] = 0.0f;
    four_dimensional[4] = 0.0f;
    four_dimensional[5] = 0.0f;
    four_dimensional[6] = -1.0f;
    four_dimensional[7] = 4.0f;
    const Tensor<float> four_dimensional_result =
        forward_pass(four_dimensional, weights, bias);
    tests.check(four_dimensional_result.get_shape() ==
                    std::vector<std::size_t>({2, 1, 2, 3}),
                "4D input replaces only its last dimension");
    tests.check(four_dimensional_result[{1, 0, 1, 0}] == 0.0f &&
                    four_dimensional_result[{1, 0, 1, 1}] == 2.0f &&
                    four_dimensional_result[{1, 0, 1, 2}] == 14.0f,
                "4D output values match the same row calculation");

    tests.expect_invalid_argument(
        [&] { (void)forward_pass(one_dimensional, Tensor<float>({2, 3, 1}), bias); },
        "weights must be 2D");
    tests.expect_invalid_argument(
        [&] { (void)forward_pass(one_dimensional, weights, Tensor<float>({1, 3})); },
        "bias must be 1D");
    tests.expect_invalid_argument(
        [&] { (void)forward_pass(one_dimensional, Tensor<float>({3, 2}), bias); },
        "weight input width must match the input");
    tests.expect_invalid_argument(
        [&] { (void)forward_pass(one_dimensional, weights, Tensor<float>({2})); },
        "bias length must match the output width");
  } catch (const std::exception &error) {
    std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
    return 1;
  } catch (...) {
    std::cerr << "FAIL: unexpected unknown exception\n";
    return 1;
  }
  return tests.result();
}
