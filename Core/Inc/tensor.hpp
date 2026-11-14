#pragma once

#include <cstddef>
#include <format>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

template <typename T> class Tensor {
public:
  Tensor(const std::vector<std::size_t> &dimension_sizes);

  template <typename U>
  friend std::ostream &operator<<(std::ostream &os, const Tensor<U> &aTensor);

  // getters
  const std::vector<std::size_t> &get_shape() const { return shape; }
  const std::vector<std::size_t> &get_strides() const { return strides; }
  size_t get_size() const { return total_size; }

  // bracket operator (vector dim)
  T &operator[](const std::vector<std::size_t> &idx);
  T operator[](const std::vector<std::size_t> &idx) const;

  // bracket operator (flat dim)
  T &operator[](std::size_t idx) { return value[idx]; }
  T operator[](std::size_t idx) const { return value[idx]; }

  // += operator
  Tensor<T> &operator+=(const Tensor<T> &otherTensor);

private:
  std::vector<std::size_t> shape;   // holds the size
  std::vector<std::size_t> strides; // how far into the storage for each idx;
  std::vector<T> value;
  size_t total_size;
};

template <typename T>
Tensor<T> operator+(const Tensor<T> &lhs, const Tensor<T> &rhs);

#include "tensor.tpp"