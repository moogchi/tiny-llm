#include "math.hpp"

#include <cmath>
#include <cstddef>
#include <exception>
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
    } catch (const std::exception &error) {
      ++failures;
      std::cerr << "FAIL: " << name << " (wrong exception: " << error.what()
                << ")\n";
    }
  }

  int result() const {
    std::cout << "Attention: " << (checks - failures) << '/' << checks
              << " checks passed\n";
    return failures == 0 ? 0 : 1;
  }

private:
  int checks{};
  int failures{};
};

bool near(float actual, float expected, float tolerance = 0.001f) {
  return std::isfinite(actual) && std::fabs(actual - expected) <= tolerance;
}

int main() {
  TestRunner tests;
  try {
    Tensor<float> q({2, 2, 1});
    Tensor<float> k({2, 2, 1});
    Tensor<float> v({2, 2, 1});
    v[{0, 0, 0}] = 2.0f;
    v[{0, 1, 0}] = 10.0f;
    v[{1, 0, 0}] = 100.0f;
    v[{1, 1, 0}] = 200.0f;
    const Tensor<float> result = causal_self_attention(q, k, v);
    tests.check(result.get_shape() == std::vector<std::size_t>({2, 2, 1}),
                "output preserves batch, sequence, and width");
    tests.check(near(result[{0, 0, 0}], 2.0f) &&
                    near(result[{0, 1, 0}], 6.0f),
                "causal mask and equal scores match hand calculation");
    tests.check(near(result[{1, 0, 0}], 100.0f) &&
                    near(result[{1, 1, 0}], 150.0f),
                "batch items attend independently");
    tests.check(q[{0, 0, 0}] == 0.0f && k[{1, 1, 0}] == 0.0f &&
                    v[{0, 1, 0}] == 10.0f,
                "attention leaves all inputs unchanged");

    Tensor<float> q2({1, 2, 2});
    Tensor<float> k2({1, 2, 2});
    Tensor<float> v2({1, 2, 2});
    q2[{0, 1, 0}] = 1.0f;
    k2[{0, 0, 0}] = 1.0f;
    k2[{0, 1, 1}] = 1.0f;
    v2[{0, 0, 0}] = 2.0f;
    v2[{0, 0, 1}] = 4.0f;
    v2[{0, 1, 0}] = 10.0f;
    v2[{0, 1, 1}] = 20.0f;
    const Tensor<float> scaled = causal_self_attention(q2, k2, v2);
    tests.check(near(scaled[{0, 0, 0}], 2.0f) &&
                    near(scaled[{0, 0, 1}], 4.0f),
                "first token sees only itself for vector values");
    tests.check(near(scaled[{0, 1, 0}], 4.6419f) &&
                    near(scaled[{0, 1, 1}], 9.2838f),
                "width-two dot product uses square-root scaling");

    Tensor<float> future_q({1, 3, 1});
    Tensor<float> future_k({1, 3, 1});
    Tensor<float> future_v({1, 3, 1});
    for (std::size_t t = 0; t < 3; ++t)
      future_q[{0, t, 0}] = 1.0f;
    future_k[{0, 2, 0}] = 1000.0f;
    future_v[{0, 0, 0}] = 3.0f;
    future_v[{0, 1, 0}] = 6.0f;
    future_v[{0, 2, 0}] = 12.0f;
    const Tensor<float> before = causal_self_attention(future_q, future_k, future_v);
    future_v[{0, 2, 0}] = -1000.0f;
    const Tensor<float> after = causal_self_attention(future_q, future_k, future_v);
    tests.check(near(before[{0, 0, 0}], 3.0f) &&
                    near(before[{0, 1, 0}], 4.5f),
                "large future key cannot influence earlier outputs");
    tests.check(near(after[{0, 0, 0}], before[{0, 0, 0}]) &&
                    near(after[{0, 1, 0}], before[{0, 1, 0}]),
                "changing future value leaves earlier outputs unchanged");

    tests.expect_invalid_argument(
        [&] { (void)causal_self_attention(Tensor<float>({2, 2}), k, v); },
        "query must be three-dimensional");
    tests.expect_invalid_argument(
        [&] { (void)causal_self_attention(q, Tensor<float>({2, 2}), v); },
        "key must be three-dimensional");
    tests.expect_invalid_argument(
        [&] { (void)causal_self_attention(q, k, Tensor<float>({2, 2})); },
        "value must be three-dimensional");
    tests.expect_invalid_argument(
        [&] { (void)causal_self_attention(q, Tensor<float>({2, 2, 2}), v); },
        "key shape must match query shape");
    tests.expect_invalid_argument(
        [&] { (void)causal_self_attention(q, k, Tensor<float>({2, 1, 1})); },
        "value shape must match query shape");
  } catch (const std::exception &error) {
    std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
    return 1;
  }
  return tests.result();
}
