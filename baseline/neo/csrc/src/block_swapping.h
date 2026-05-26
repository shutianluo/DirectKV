#pragma once

#include <torch/extension.h>

#include <vector>
#include <cstdint>

void swap_blocks(
	const std::vector<int64_t> &source_block_ids,
	const std::vector<int64_t> &target_block_ids,
	const bool is_swap_out,
	const int gpu_layer,
	const int cpu_layer,

	torch::Tensor k_cache,
	torch::Tensor v_cache,
	torch::Tensor k_swap,
	torch::Tensor v_swap
);
