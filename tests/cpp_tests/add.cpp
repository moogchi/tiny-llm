#include "tensor.hpp"

#include <iostream>
#include <stdexcept>
#include <string_view>

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
    std::cout << "Addition: " << (checks - failures) << '/' << checks
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
    Tensor<float> left({2, 2});
    left[{0, 0}] = 1.0f;
    left[{0, 1}] = -2.0f;
    left[{1, 0}] = 0.5f;
    left[{1, 1}] = 4.0f;
    Tensor<float> right({2, 2});
    right[{0, 0}] = 3.0f;
    right[{0, 1}] = 2.0f;
    right[{1, 0}] = -1.5f;
    right[{1, 1}] = 6.0f;

    const Tensor<float> const_left = left;
    const Tensor<float> const_right = right;
    const Tensor<float> sum = const_left + const_right;
    tests.check(sum.get_shape() == left.get_shape(),
                "plus preserves the full shape");
    tests.check(sum[{0, 0}] == 4.0f && sum[{0, 1}] == 0.0f &&
                    sum[{1, 0}] == -1.0f && sum[{1, 1}] == 10.0f,
                "plus adds corresponding values");
    tests.check(left[{0, 0}] == 1.0f && right[{0, 0}] == 3.0f,
                "plus leaves both inputs unchanged");

    Tensor<float> &returned = (left += right);
    tests.check(&returned == &left, "plus-equals returns the left tensor");
    tests.check(left[{0, 0}] == 4.0f && left[{0, 1}] == 0.0f &&
                    left[{1, 0}] == -1.0f && left[{1, 1}] == 10.0f,
                "plus-equals updates all values");
    tests.check(right[{0, 0}] == 3.0f && right[{1, 1}] == 6.0f,
                "plus-equals leaves the right tensor unchanged");

    Tensor<float> mismatched_left({2, 3});
    mismatched_left[{0, 0}] = 7.0f;
    Tensor<float> mismatched_right({3, 2});
    tests.expect_invalid_argument(
        [&] { mismatched_left += mismatched_right; },
        "plus-equals rejects different shapes with equal element counts");
    tests.check(mismatched_left[{0, 0}] == 7.0f &&
                    mismatched_left.get_shape().size() == 2 &&
                    mismatched_left.get_shape()[0] == 2,
                "failed plus-equals leaves the left tensor unchanged");
    tests.expect_invalid_argument(
        [&] { (void)(mismatched_left + mismatched_right); },
        "plus rejects different shapes with equal element counts");
    tests.check(mismatched_left[{0, 0}] == 7.0f,
                "failed plus leaves its inputs unchanged");
  } catch (const std::exception &error) {
    std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
    return 1;
  } catch (...) {
    std::cerr << "FAIL: unexpected unknown exception\n";
    return 1;
  }
  return tests.result();
}
