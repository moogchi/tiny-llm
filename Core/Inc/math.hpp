#pragma once
#include "tensor.hpp"

#include <algorithm>
#include <cmath>

Tensor<float> matmul(const Tensor<float> &aTensor,
                     const Tensor<float> &bTensor);

Tensor<float> softmax(const Tensor<float> &aTensor);

Tensor<float> layer_normalization(const Tensor<float> &aTensor);

Tensor<float> forward_pass(const Tensor<float> &input,
                           const Tensor<float> &weights,
                           const Tensor<float> &bias);

Tensor<float> gelu(const Tensor<float> &aTensor);

Tensor<float> embedding_layer(const Tensor<std::size_t> &input,
                              const Tensor<float> &table);

Tensor<float> causal_self_attention(const Tensor<float> &Q,
                                    const Tensor<float> &K,
                                    const Tensor<float> &V);