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
    std::cout << "GELU: " << (checks - failures) << '/' << checks
              << " checks passed\n";
    return failures == 0 ? 0 : 1;
  }

private:
  int checks{};
  int failures{};
};

bool close_to(float actual, float expected, float tolerance = 0.00001f) {
  return std::isfinite(actual) && std::fabs(actual - expected) <= tolerance;
}

int main() {
  TestRunner tests;
  try {
    Tensor<float> values({7});
    values[0] = -10.0f;
    values[1] = -2.0f;
    values[2] = -1.0f;
    values[3] = 0.0f;
    values[4] = 1.0f;
    values[5] = 2.0f;
    values[6] = 10.0f;
    const Tensor<float> result = gelu(values);
    tests.check(result.get_shape() == std::vector<std::size_t>{7},
                "1D output keeps its shape");
    tests.check(close_to(result[0], 0.0f),
                "large negative input approaches zero");
    tests.check(close_to(result[1], -0.0455003f),
                "GELU of negative two matches the exact reference");
    tests.check(close_to(result[2], -0.1586553f),
                "GELU of negative one matches the exact reference");
    tests.check(close_to(result[3], 0.0f), "GELU of zero is zero");
    tests.check(close_to(result[4], 0.8413447f),
                "GELU of positive one matches the exact reference");
    tests.check(close_to(result[5], 1.9544997f),
                "GELU of positive two matches the exact reference");
    tests.check(close_to(result[6], 10.0f),
                "large positive input approaches the input");
    tests.check(values[1] == -2.0f && values[5] == 2.0f,
                "GELU leaves its input unchanged");

    Tensor<float> three_dimensional({2, 2, 2});
    three_dimensional[0] = -1.0f;
    three_dimensional[1] = 0.0f;
    three_dimensional[2] = 1.0f;
    three_dimensional[3] = 2.0f;
    three_dimensional[4] = -2.0f;
    three_dimensional[5] = 3.0f;
    three_dimensional[6] = 0.0f;
    three_dimensional[7] = 1.0f;
    const Tensor<float> three_dimensional_result = gelu(three_dimensional);
    tests.check(three_dimensional_result.get_shape() ==
                    std::vector<std::size_t>({2, 2, 2}),
                "3D output keeps every dimension");
    tests.check(close_to(three_dimensional_result[{0, 0, 0}], -0.1586553f) &&
                    close_to(three_dimensional_result[{0, 1, 0}], 0.8413447f) &&
                    close_to(three_dimensional_result[{1, 0, 0}], -0.0455003f) &&
                    close_to(three_dimensional_result[{1, 1, 1}], 0.8413447f),
                "3D values are transformed independently");

    Tensor<float> four_dimensional({1, 2, 1, 2});
    four_dimensional[0] = -1.0f;
    four_dimensional[1] = 1.0f;
    four_dimensional[2] = 2.0f;
    four_dimensional[3] = -2.0f;
    const Tensor<float> four_dimensional_result = gelu(four_dimensional);
    tests.check(four_dimensional_result.get_shape() ==
                    std::vector<std::size_t>({1, 2, 1, 2}),
                "4D output keeps every dimension");
    tests.check(close_to(four_dimensional_result[0], -0.1586553f) &&
                    close_to(four_dimensional_result[1], 0.8413447f) &&
                    close_to(four_dimensional_result[2], 1.9544997f) &&
                    close_to(four_dimensional_result[3], -0.0455003f),
                "4D values follow the same scalar rule");
  } catch (const std::exception &error) {
    std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
    return 1;
  } catch (...) {
    std::cerr << "FAIL: unexpected unknown exception\n";
    return 1;
  }
  return tests.result();
}
