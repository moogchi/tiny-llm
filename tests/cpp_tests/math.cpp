#include "math.hpp"
#include "tensor.hpp"

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
    std::cout << "Matmul: " << (checks - failures) << '/' << checks
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
    Tensor<float> left({2, 3});
    left[{0, 0}] = 1.0f;
    left[{0, 1}] = 2.0f;
    left[{0, 2}] = 3.0f;
    left[{1, 0}] = 4.0f;
    left[{1, 1}] = 5.0f;
    left[{1, 2}] = 6.0f;

    Tensor<float> right({3, 4});
    right[{0, 0}] = 1.0f;
    right[{0, 1}] = 2.0f;
    right[{0, 2}] = 3.0f;
    right[{0, 3}] = 4.0f;
    right[{1, 0}] = 5.0f;
    right[{1, 1}] = 6.0f;
    right[{1, 2}] = 7.0f;
    right[{1, 3}] = 8.0f;
    right[{2, 0}] = 9.0f;
    right[{2, 1}] = 10.0f;
    right[{2, 2}] = 11.0f;
    right[{2, 3}] = 12.0f;

    const Tensor<float> product = matmul(left, right);
    tests.check(product.get_shape() == std::vector<std::size_t>({2, 4}),
                "2x3 times 3x4 produces a 2x4 shape");
    tests.check(product.get_size() == 8, "2x4 product has eight elements");
    tests.check(product[{0, 0}] == 38.0f, "product row 0 column 0");
    tests.check(product[{0, 1}] == 44.0f, "product row 0 column 1");
    tests.check(product[{0, 2}] == 50.0f, "product row 0 column 2");
    tests.check(product[{0, 3}] == 56.0f, "product row 0 column 3");
    tests.check(product[{1, 0}] == 83.0f, "product row 1 column 0");
    tests.check(product[{1, 1}] == 98.0f, "product row 1 column 1");
    tests.check(product[{1, 2}] == 113.0f, "product row 1 column 2");
    tests.check(product[{1, 3}] == 128.0f, "product row 1 column 3");
    tests.check(left[{1, 2}] == 6.0f && right[{2, 3}] == 12.0f,
                "matmul does not change either input");

    Tensor<float> negative_left({1, 2});
    negative_left[{0, 0}] = -2.0f;
    negative_left[{0, 1}] = 3.0f;
    Tensor<float> negative_right({2, 1});
    negative_right[{0, 0}] = 4.0f;
    negative_right[{1, 0}] = 5.0f;
    const Tensor<float> signed_product = matmul(negative_left, negative_right);
    tests.check(signed_product.get_shape() == std::vector<std::size_t>({1, 1}),
                "1x2 times 2x1 produces a 1x1 shape");
    tests.check(signed_product[{0, 0}] == 7.0f,
                "dot product handles negative values");

    tests.expect_invalid_argument(
        [&left] { (void)matmul(left, Tensor<float>({4, 2})); },
        "mismatched inner dimensions are rejected");
    tests.expect_invalid_argument(
        [&right] { (void)matmul(Tensor<float>({2, 3, 1}), right); },
        "left input must be 2D");
    tests.expect_invalid_argument(
        [&left] { (void)matmul(left, Tensor<float>({3})); },
        "right input must be 2D");
  } catch (const std::exception &error) {
    std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
    return 1;
  } catch (...) {
    std::cerr << "FAIL: unexpected unknown exception\n";
    return 1;
  }
  return tests.result();
}
