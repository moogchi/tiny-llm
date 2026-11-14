#include "tensor.hpp"
#include <cstddef>
#include <limits>
#include <stdexcept>

template <typename T>
Tensor<T>::Tensor(const std::vector<std::size_t> &dimension_sizes) {
  if (dimension_sizes.size() == 0)
    throw std::invalid_argument("Not a valid dimension");

  for (std::size_t idx{0}; idx < dimension_sizes.size(); ++idx) {
    if (dimension_sizes[idx] == 0)
      throw std::invalid_argument(
          std::format("Zero Dimention found in index: {}", idx));
  }

  shape = dimension_sizes; // copy the shape

  strides.assign(shape.size(), 1); // have a stride of same lenght as shape

  // look at right side and shape idx;
  for (std::size_t idx{shape.size() - 1}; idx > 0; --idx) {
    // overflow check
    std::size_t multiplicand{strides[idx]}, multiplier{shape[idx]};
    if (multiplicand > (std::numeric_limits<std::size_t>::max() / multiplier))
      throw std::overflow_error("Dimension is too big");

    strides[idx - 1] = multiplicand * multiplier;
  }

  // calculate total size
  // overflow check
  if (strides[0] > (std::numeric_limits<std::size_t>::max() / shape[0]))
    throw std::overflow_error("Dimension is too big");
  total_size = strides[0] * shape[0];
  // zero vector
  value.assign(total_size, static_cast<T>(0));
}

template <typename T>
T &Tensor<T>::operator[](const std::vector<std::size_t> &idx) {
  // check if idx is the same size as shape
  if (idx.size() != shape.size())
    throw std::out_of_range("Idx is not the correct size.");

  std::size_t true_idx{};
  for (std::size_t i{}; i < idx.size(); ++i) {
    // check if that index exists
    if (idx[i] >= shape[i])
      throw std::out_of_range(
          std::format("Index is not valid at Dimension {}", i));

    // calculate idx based on stride
    true_idx += idx[i] * strides[i];
  }
  return value[true_idx];
}

template <typename T>
T Tensor<T>::operator[](const std::vector<std::size_t> &idx) const {
  // check if idx is the same size as shape
  if (idx.size() != shape.size())
    throw std::out_of_range("Idx is not the correct size.");

  std::size_t true_idx{};
  for (std::size_t i{}; i < idx.size(); ++i) {
    // check if that index exists
    if (idx[i] >= shape[i])
      throw std::out_of_range(
          std::format("Index is not valid at Dimension {}", i));

    // calculate idx based on stride
    true_idx += idx[i] * strides[i];
  }
  return value[true_idx];
}

template <typename T> Tensor<T> &Tensor<T>::operator+=(const Tensor &rhs) {
  // check if they are the same size
  if (shape != rhs.shape) {
    throw std::invalid_argument("Shape of Tensor is Different");
  }

  // add the values
  for (size_t idx{}; idx < value.size(); ++idx) {
    value[idx] += rhs.value[idx];
  }
  // return the current tensor
  return *this;
}

template <typename T>
Tensor<T> operator+(const Tensor<T> &lhs, const Tensor<T> &rhs) {
  Tensor<T> aTensor(lhs);
  aTensor += rhs;
  return aTensor;
}

template <typename U>
std::ostream &operator<<(std::ostream &os, const Tensor<U> &aTensor) {
  os << "Shape: ";
  for (size_t idx{}; idx < aTensor.shape.size(); ++idx) {
    os << aTensor.shape[idx] << ", ";
  }
  os << std::endl;

  os << "Strides: ";
  for (size_t idx{}; idx < aTensor.strides.size(); ++idx) {
    os << aTensor.strides[idx] << ", ";
  }
  os << std::endl;

  os << "Values: ";
  for (size_t idx{}; idx < aTensor.value.size(); ++idx) {
    os << aTensor.value[idx] << ", ";
  }
  os << std::endl;
  return os;
}