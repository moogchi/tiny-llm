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
    std::cout << "Embedding: " << (checks - failures) << '/' << checks
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
    Tensor<float> table({4, 3});
    table[{0, 0}] = 0.1f;
    table[{0, 1}] = 0.2f;
    table[{0, 2}] = 0.3f;
    table[{1, 0}] = -1.0f;
    table[{1, 1}] = -2.0f;
    table[{1, 2}] = -3.0f;
    table[{2, 0}] = 2.0f;
    table[{2, 1}] = 4.0f;
    table[{2, 2}] = 6.0f;
    table[{3, 0}] = 7.0f;
    table[{3, 1}] = 8.0f;
    table[{3, 2}] = 9.0f;

    Tensor<std::size_t> ids({2, 3});
    ids[{0, 0}] = 2;
    ids[{0, 1}] = 0;
    ids[{0, 2}] = 2;
    ids[{1, 0}] = 3;
    ids[{1, 1}] = 1;
    ids[{1, 2}] = 0;
    const Tensor<float> result = embedding_layer(ids, table);
    tests.check(result.get_shape() == std::vector<std::size_t>({2, 3, 3}),
                "output shape is batch, sequence, embedding width");
    tests.check(result[{0, 0, 0}] == 2.0f && result[{0, 0, 1}] == 4.0f &&
                    result[{0, 0, 2}] == 6.0f,
                "first token selects its exact table row");
    tests.check(result[{0, 1, 0}] == 0.1f && result[{0, 1, 1}] == 0.2f &&
                    result[{0, 1, 2}] == 0.3f,
                "token zero selects the first row");
    tests.check(result[{1, 0, 0}] == 7.0f && result[{1, 0, 1}] == 8.0f &&
                    result[{1, 0, 2}] == 9.0f,
                "largest valid ID selects the last row");
    tests.check(result[{1, 1, 0}] == -1.0f && result[{1, 1, 1}] == -2.0f &&
                    result[{1, 1, 2}] == -3.0f,
                "second batch selects the correct row");
    tests.check(result[{0, 2, 0}] == result[{0, 0, 0}] &&
                    result[{0, 2, 1}] == result[{0, 0, 1}] &&
                    result[{0, 2, 2}] == result[{0, 0, 2}],
                "repeated IDs produce equal vectors");
    tests.check(table[{2, 0}] == 2.0f && ids[{0, 0}] == 2,
                "lookup leaves table and IDs unchanged");
    Tensor<float> changed = result;
    changed[{0, 0, 0}] = 99.0f;
    tests.check(table[{2, 0}] == 2.0f && result[{0, 0, 0}] == 2.0f,
                "output values are independent of the table");

    Tensor<std::size_t> invalid_id({1, 1});
    invalid_id[{0, 0}] = 4;
    tests.expect_exception<std::out_of_range>(
        [&] { (void)embedding_layer(invalid_id, table); },
        "ID equal to vocabulary size is rejected");
    tests.expect_exception<std::invalid_argument>(
        [&] { (void)embedding_layer(ids, Tensor<float>({4, 3, 1})); },
        "embedding table must be 2D");
    tests.expect_exception<std::invalid_argument>(
        [&] { (void)embedding_layer(Tensor<std::size_t>({3}), table); },
        "ID input must be 2D");
    tests.expect_exception<std::invalid_argument>(
        [&] { (void)embedding_layer(Tensor<std::size_t>({1, 2, 3}), table); },
        "3D ID input is rejected");
  } catch (const std::exception &error) {
    std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
    return 1;
  } catch (...) {
    std::cerr << "FAIL: unexpected unknown exception\n";
    return 1;
  }
  return tests.result();
}
