#pragma once

#include <torch/extension.h>

torch::Tensor linear(
  torch::Tensor a,
  torch::Tensor w
);

void linear_inplace(
  torch::Tensor a,
  torch::Tensor w,
  torch::Tensor r
);