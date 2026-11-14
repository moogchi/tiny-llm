#include "math.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
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

  int result() const {
    std::cout << "Softmax: " << (checks - failures) << '/' << checks
              << " checks passed\n";
    return failures == 0 ? 0 : 1;
  }

private:
  int checks{};
  int failures{};
};

bool close_to(float actual, float expected, float tolerance = 0.0001f) {
  return std::isfinite(actual) && std::fabs(actual - expected) <= tolerance;
}

int main() {
  TestRunner tests;
  try {
    Tensor<float> one_dimensional({3});
    one_dimensional[0] = 1.0f;
    one_dimensional[1] = 2.0f;
    one_dimensional[2] = 3.0f;
    const Tensor<float> one_dimensional_result = softmax(one_dimensional);
    tests.check(one_dimensional_result.get_shape() ==
                    std::vector<std::size_t>{3},
                "1D output keeps its shape");
    tests.check(close_to(one_dimensional_result[0], 0.0900306f) &&
                    close_to(one_dimensional_result[1], 0.2447285f) &&
                    close_to(one_dimensional_result[2], 0.6652409f),
                "1D values match a known reference");
    tests.check(close_to(one_dimensional_result[0] +
                             one_dimensional_result[1] +
                             one_dimensional_result[2],
                         1.0f),
                "1D probabilities sum to one");
    tests.check(one_dimensional[0] == 1.0f && one_dimensional[2] == 3.0f,
                "softmax does not change its input");

    Tensor<float> two_dimensional({2, 3});
    two_dimensional[{0, 0}] = 0.0f;
    two_dimensional[{0, 1}] = 0.0f;
    two_dimensional[{0, 2}] = 0.0f;
    two_dimensional[{1, 0}] = 1000.0f;
    two_dimensional[{1, 1}] = 1001.0f;
    two_dimensional[{1, 2}] = 1002.0f;
    const Tensor<float> two_dimensional_result = softmax(two_dimensional);
    tests.check(two_dimensional_result.get_shape() ==
                    std::vector<std::size_t>({2, 3}),
                "2D output keeps its shape");
    tests.check(close_to(two_dimensional_result[{0, 0}], 1.0f / 3.0f) &&
                    close_to(two_dimensional_result[{0, 1}], 1.0f / 3.0f) &&
                    close_to(two_dimensional_result[{0, 2}], 1.0f / 3.0f),
                "equal logits give a uniform row");
    tests.check(close_to(two_dimensional_result[{1, 0}], 0.0900306f) &&
                    close_to(two_dimensional_result[{1, 1}], 0.2447285f) &&
                    close_to(two_dimensional_result[{1, 2}], 0.6652409f),
                "large positive logits stay stable and rows stay independent");

    Tensor<float> three_dimensional({2, 2, 2});
    three_dimensional[{0, 0, 0}] = -1000.0f;
    three_dimensional[{0, 0, 1}] = -1001.0f;
    three_dimensional[{0, 1, 0}] = 5.0f;
    three_dimensional[{0, 1, 1}] = 5.0f;
    three_dimensional[{1, 0, 0}] = 0.0f;
    three_dimensional[{1, 0, 1}] = 2.0f;
    three_dimensional[{1, 1, 0}] = -2.0f;
    three_dimensional[{1, 1, 1}] = -2.0f;
    const Tensor<float> three_dimensional_result = softmax(three_dimensional);
    tests.check(three_dimensional_result.get_shape() ==
                    std::vector<std::size_t>({2, 2, 2}),
                "3D output keeps its shape");
    tests.check(close_to(three_dimensional_result[{0, 0, 0}], 0.7310586f) &&
                    close_to(three_dimensional_result[{0, 0, 1}], 0.2689414f),
                "large negative logits stay stable");
    tests.check(close_to(three_dimensional_result[{0, 1, 0}], 0.5f) &&
                    close_to(three_dimensional_result[{0, 1, 1}], 0.5f) &&
                    close_to(three_dimensional_result[{1, 1, 0}], 0.5f) &&
                    close_to(three_dimensional_result[{1, 1, 1}], 0.5f),
                "3D rows are normalized independently");
    bool all_rows_sum_to_one = true;
    for (std::size_t row = 0; row < 4; ++row) {
      all_rows_sum_to_one =
          all_rows_sum_to_one &&
          close_to(three_dimensional_result[row * 2] +
                       three_dimensional_result[row * 2 + 1],
                   1.0f);
    }
    tests.check(all_rows_sum_to_one, "every 3D row sums to one");

    Tensor<float> four_dimensional({2, 1, 2, 1});
    four_dimensional[0] = -2000.0f;
    four_dimensional[1] = 7.0f;
    four_dimensional[2] = 0.0f;
    four_dimensional[3] = 2000.0f;
    const Tensor<float> four_dimensional_result = softmax(four_dimensional);
    tests.check(four_dimensional_result.get_shape() ==
                    std::vector<std::size_t>({2, 1, 2, 1}),
                "4D output keeps its shape");
    tests.check(close_to(four_dimensional_result[0], 1.0f) &&
                    close_to(four_dimensional_result[1], 1.0f) &&
                    close_to(four_dimensional_result[2], 1.0f) &&
                    close_to(four_dimensional_result[3], 1.0f),
                "last dimension of length one always yields one");
  } catch (const std::exception &error) {
    std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
    return 1;
  } catch (...) {
    std::cerr << "FAIL: unexpected unknown exception\n";
    return 1;
  }
  return tests.result();
}
