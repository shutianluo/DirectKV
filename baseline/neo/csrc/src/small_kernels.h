# pragma once

#include <torch/extension.h>
#include <cstdint>
#include <cuda_fp16.h>

void fused_add_rmsnorm_inplace(
    torch::Tensor buffer,
    torch::Tensor residual,
    torch::Tensor weight,
    const float epsilon
);

void silu_and_mul_inplace(
    torch::Tensor buffer
);

void rotary_embedding_inplace(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor sin_table,
    torch::Tensor cos_table
);

void store_kvcache(
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_table,
    torch::Tensor seq_ids,
    torch::Tensor seq_start_locs,
    torch::Tensor seq_lens,
    const int64_t itm_layer,
    const int64_t gpu_layer,
    const int64_t num_cprfs,
    const int64_t max_pref_len
);

void embedding(
    torch::Tensor input_tokens,
    torch::Tensor weights,
    torch::Tensor output,
    const int64_t token_offset
);