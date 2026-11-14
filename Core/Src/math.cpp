#include "math.hpp"
#include "tensor.hpp"
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

// 2D Tensor (Matrix) Multiplication (CPU Implementation)
Tensor<float> matmul(const Tensor<float> &aTensor,
                     const Tensor<float> &bTensor) {
  // check if both Tensors are 2d.
  if (aTensor.get_shape().size() != 2 || bTensor.get_shape().size() != 2) {
    throw std::invalid_argument("Tensors are not 2D. ");
  }

  // check if Tensors are compatible for multiplication
  std::size_t aTensorInnDim = aTensor.get_shape()[1];
  std::size_t bTensorInnDim = bTensor.get_shape()[0];
  if (aTensorInnDim != bTensorInnDim) {
    throw std::invalid_argument("Tensor inner dimensions differ");
  }

  // Initialize new 2d Tensor with Outer Dimensions
  std::size_t aTensorOutDim = aTensor.get_shape()[0];
  std::size_t bTensorOutDim = bTensor.get_shape()[1];
  std::vector<std::size_t> dim = {aTensorOutDim, bTensorOutDim};
  Tensor<float> retTensor(dim);

  // Multiplication
  for (std::size_t i{}; i < aTensorOutDim; ++i) {
    for (std::size_t j{}; j < aTensorInnDim; ++j) {
      for (std::size_t k{}; k < bTensorOutDim; ++k) {
        retTensor[{i, k}] += aTensor[{i, j}] * bTensor[{j, k}];
      }
    }
  }
  return retTensor;
}

Tensor<float> softmax(const Tensor<float> &aTensor) {
  Tensor<float> retTensor(aTensor);
  // get last element of Tensor's shape
  std::size_t batch_size = aTensor.get_shape()[aTensor.get_shape().size() - 1];

  // find max
  std::size_t batch_quantity = aTensor.get_size() / batch_size;
  std::vector<float> batch_max(batch_quantity, 0);

  for (std::size_t batch_idx{}; batch_idx < batch_quantity; ++batch_idx) {
    for (std::size_t inter_idx{}; inter_idx < batch_size; ++inter_idx) {
      std::size_t flat_idx = batch_idx * batch_size + inter_idx;
      if (inter_idx == 0)
        batch_max[batch_idx] = aTensor[flat_idx];
      batch_max[batch_idx] = std::fmax(batch_max[batch_idx], aTensor[flat_idx]);
    }
  }

  // subtract max and also make it a power of e
  std::vector<float> batch_sum(batch_quantity, 0);
  for (std::size_t batch_idx{}; batch_idx < batch_quantity; ++batch_idx) {
    for (std::size_t inter_idx{}; inter_idx < batch_size; ++inter_idx) {
      std::size_t flat_idx = batch_idx * batch_size + inter_idx;
      retTensor[flat_idx] = std::exp(aTensor[flat_idx] - batch_max[batch_idx]);
      batch_sum[batch_idx] += retTensor[flat_idx];
    }
  }

  // divide by batch_sum to make it add up to 1
  for (std::size_t batch_idx{}; batch_idx < batch_quantity; ++batch_idx) {
    for (std::size_t inter_idx{}; inter_idx < batch_size; ++inter_idx) {
      std::size_t flat_idx = batch_idx * batch_size + inter_idx;
      retTensor[flat_idx] /= batch_sum[batch_idx];
    }
  }

  return retTensor;
}

Tensor<float> layer_normalization(const Tensor<float> &aTensor) {
  Tensor<float> retTensor(aTensor);

  // get last element of Tensor's shape
  std::size_t batch_size = aTensor.get_shape()[aTensor.get_shape().size() - 1];

  // find max
  std::size_t batch_quantity = aTensor.get_size() / batch_size;
  std::vector<float> batch_mean(batch_quantity, 0);

  for (std::size_t batch_idx{}; batch_idx < batch_quantity; ++batch_idx) {
    for (std::size_t inter_idx{}; inter_idx < batch_size; ++inter_idx) {
      std::size_t flat_idx = batch_idx * batch_size + inter_idx;
      batch_mean[batch_idx] += aTensor[flat_idx];
    }
    batch_mean[batch_idx] /= static_cast<float>(batch_size);
  }

  // find variance
  std::vector<float> batch_variance(batch_quantity, 0);
  for (std::size_t batch_idx{}; batch_idx < batch_quantity; ++batch_idx) {
    for (std::size_t inter_idx{}; inter_idx < batch_size; ++inter_idx) {
      std::size_t flat_idx = batch_idx * batch_size + inter_idx;
      batch_variance[batch_idx] +=
          std::pow(aTensor[flat_idx] - batch_mean[batch_idx], 2.0f);
    }
    batch_variance[batch_idx] /= static_cast<float>(batch_size);
  }

  // normalize each value
  constexpr float epsilon = 0.00001f; // prevents div by zero
  for (std::size_t batch_idx{}; batch_idx < batch_quantity; ++batch_idx) {
    for (std::size_t inter_idx{}; inter_idx < batch_size; ++inter_idx) {
      std::size_t flat_idx = batch_idx * batch_size + inter_idx;
      retTensor[flat_idx] =
          (aTensor[flat_idx] - batch_mean[batch_idx]) /
          (std::sqrt(static_cast<float>(batch_variance[batch_idx]) + epsilon));
    }
  }

  return retTensor;
}

Tensor<float> forward_pass(const Tensor<float> &input,
                           const Tensor<float> &weights,
                           const Tensor<float> &bias) {
  // weights need to be 2d
  if (weights.get_shape().size() != 2)
    throw std::invalid_argument("Weight isn't 2d Tensor.");

  // bias has to be 1d tensor
  if (bias.get_shape().size() != 1)
    throw std::invalid_argument("Bias isn't 1d Tensor");

  // get batch size and quantity based on the last elements
  std::size_t batch_size = input.get_shape()[input.get_shape().size() - 1];
  std::size_t batch_quantity = input.get_size() / batch_size;

  // verify that weights are the correct shape.
  if (weights.get_shape()[0] != batch_size)
    throw std::invalid_argument("Weight's # of rows does not match input");

  // verify that bias has are the correct length
  if (bias.get_shape()[0] != weights.get_shape()[1])
    throw std::invalid_argument("Weight's # of columns does not match bias");

  // Initialize output with dimension input{0...last-1, bias}.
  std::vector<std::size_t> output_dim = input.get_shape();
  output_dim[output_dim.size() - 1] = bias.get_shape()[0];
  Tensor<float> output(output_dim);

  // calculation Tensor, modifies any dimension tensor to 2d
  Tensor<float> inputCalculationTensor({batch_quantity, batch_size});
  for (std::size_t idx{}; idx < input.get_size(); ++idx) {
    inputCalculationTensor[idx] = input[idx];
  }

  // multiply the 2d vectors
  Tensor<float> outputCalculationTensor({batch_quantity, bias.get_shape()[0]});
  outputCalculationTensor = matmul(inputCalculationTensor, weights);

  for (std::size_t batch_idx{}; batch_idx < batch_quantity; ++batch_idx) {
    for (std::size_t bias_idx{}; bias_idx < bias.get_shape()[0]; ++bias_idx) {
      outputCalculationTensor[{batch_idx, bias_idx}] += bias[bias_idx];
    }
  }

  // convert the output calculation tensor back into proper tensor shape
  for (std::size_t idx{}; idx < output.get_size(); ++idx) {
    output[idx] = outputCalculationTensor[idx];
  }

  return output;
}

Tensor<float> gelu(const Tensor<float> &aTensor) {
  Tensor<float> retTensor(aTensor);
  for (std::size_t idx{}; idx < aTensor.get_size(); ++idx) {
    float x = aTensor[idx];
    float gelu_val = x / 2 * (1 + std::erf(x / std::sqrt(2.0f)));
    retTensor[idx] = gelu_val;
  }
  return retTensor;
}

Tensor<float> embedding_layer(const Tensor<std::size_t> &input,
                              const Tensor<float> &table) {

  if (table.get_shape().size() != 2)
    throw std::invalid_argument("Table is not a 2d tensor");

  if (input.get_shape().size() != 2)
    throw std::invalid_argument("Input is not a 2d tensor");

  // output tensor
  std::vector<std::size_t> output_dim = {
      input.get_shape()[0], input.get_shape()[1], table.get_shape()[1]};
  Tensor<float> retTensor(output_dim);

  for (std::size_t inputIdx1{}; inputIdx1 < input.get_shape()[0]; ++inputIdx1) {
    for (std::size_t inputIdx2{}; inputIdx2 < input.get_shape()[1];
         ++inputIdx2) {
      for (std::size_t tableIdx{}; tableIdx < table.get_shape()[1];
           ++tableIdx) {
        std::vector<std::size_t> flatOutputIdx{inputIdx1, inputIdx2, tableIdx};
        retTensor[(flatOutputIdx)] =
            table[{input[{inputIdx1, inputIdx2}], tableIdx}];
      }
    }
  }
  return retTensor;
}

Tensor<float> causal_self_attention(const Tensor<float> &Q,
                                    const Tensor<float> &K,
                                    const Tensor<float> &V) {
  // check if they are 3d tensor
  if (Q.get_shape().size() != 3)
    throw std::invalid_argument("Q is not a 3D Tensor");
  if (K.get_shape().size() != 3)
    throw std::invalid_argument("K is not a 3D Tensor");
  if (V.get_shape().size() != 3)
    throw std::invalid_argument("V is not a 3D Tensor");

  // check so they all have the same shape
  if (!(Q.get_shape() == K.get_shape() && K.get_shape() == V.get_shape()))
    throw std::invalid_argument("The Tensors are not of same shape");

  std::size_t b_max = Q.get_shape()[0], t_max = Q.get_shape()[1],
              d_max = Q.get_shape()[2];

  Tensor<float> score({b_max, t_max, t_max});
  // per batch math
  for (std::size_t batch{}; batch < b_max; ++batch) {
    for (std::size_t current_pos{}; current_pos < t_max; ++current_pos) {
      for (std::size_t source_pos{}; source_pos < t_max; ++source_pos) {
        float sum{};
        if (current_pos < source_pos) {
          sum = -std::numeric_limits<float>::infinity();
        } else {
          for (std::size_t vector_element{}; vector_element < d_max;
               ++vector_element) {
            sum += Q[{batch, current_pos, vector_element}] *
                   K[{batch, source_pos, vector_element}];
          }
          sum /= std::sqrt(static_cast<float>(d_max));
        }
        score[{batch, current_pos, source_pos}] = sum;
      }
    }
  }

  // run through softmax
  Tensor<float> weights = softmax(score);

  Tensor<float> output({b_max, t_max, d_max});
  for (std::size_t batch{}; batch < b_max; ++batch) {
    for (std::size_t current_pos{}; current_pos < t_max; ++current_pos) {
      for (std::size_t source_pos{}; source_pos < t_max; ++source_pos) {
        for (std::size_t vector_element{}; vector_element < d_max;
             ++vector_element) {
          output[{batch, current_pos, vector_element}] +=
              weights[{batch, current_pos, source_pos}] *
              V[{batch, source_pos, vector_element}];
        }
      }
    }
  }

  return output;
}