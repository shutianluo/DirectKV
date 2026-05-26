#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <cuda_fp16.h>

void paged_attention(
  torch::Tensor q,
  torch::Tensor k,
  torch::Tensor v,
  torch::Tensor o,
  torch::Tensor kcache,
  torch::Tensor vcache,
  float softmax_scale,
  torch::Tensor block_table,
  torch::Tensor seq_ids,
  torch::Tensor seq_lens,  
  const int64_t cur_layer,
  const int64_t seq_block_size,
  const int64_t num_seq_blocks
);