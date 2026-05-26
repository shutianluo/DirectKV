#include "block_swapping.h"

#include <ATen/cuda/CUDAContext.h>	// for at::cuda::getCurrentCUDAStream()

inline size_t getTensorSizeInBytes(const torch::Tensor &tensor) {
    return tensor.numel() * torch::elementSize(torch::typeMetaToScalarType(tensor.dtype()));
}

// swap_blocks - Perform swapping between GPU blocks and CPU blocks
// The source_block_ids and target_block_ids are the block ids of the blocks to be swapped.
// source_block_ids[0] will be copied to target_block_ids[0] and so on
// `is_swap_out` defines whether the swap is a swap-in or swap-out (swap-in means
// to swap from CPU to GPU, swap-out means to swap from GPU to CPU)
//
// Here we do not pass a cudaStream to the function. Instead we use the current
// stream indicated by at::cuda::getCurrentCUDAStream(). So it is python's
// responsibility to set the current stream before calling this function.
// 
// Future work: Now the number of cudaMemcpyAsync calls is equal to 2x the number
// of blocks to swap. We can reduce the number of cudaMemcpyAsync calls by
// grouping nearby blocks together and perform a single invocation
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
) {
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	size_t gpu_layer_size_in_bytes = getTensorSizeInBytes(k_cache) / k_cache.size(0);
	size_t cpu_layer_size_in_bytes = getTensorSizeInBytes(k_swap) / k_swap.size(0);
	size_t block_layer_size_in_bytes = gpu_layer_size_in_bytes / k_cache.size(1); // same for gpu and cpu

	char* k_cache_ptr = (char*)k_cache.data_ptr() + gpu_layer * gpu_layer_size_in_bytes;
	char* v_cache_ptr = (char*)v_cache.data_ptr() + gpu_layer * gpu_layer_size_in_bytes;
	char* k_swap_ptr = (char*)k_swap.data_ptr() + cpu_layer * cpu_layer_size_in_bytes;
	char* v_swap_ptr = (char*)v_swap.data_ptr() + cpu_layer * cpu_layer_size_in_bytes;
	int num_blocks_to_swap = source_block_ids.size();
	int next_index = 0;
	while (next_index < num_blocks_to_swap) {
		int start_index = next_index;
		int end_index = start_index+1;
		while (end_index < num_blocks_to_swap &&
					source_block_ids[end_index] == source_block_ids[end_index-1]+1 &&
					target_block_ids[end_index] == target_block_ids[end_index-1]+1) {
			end_index++;
		}
		int cur_segment_len = end_index - start_index;
		size_t cur_segment_size_in_bytes = cur_segment_len * block_layer_size_in_bytes;
		int64_t start_source_block_id = source_block_ids[start_index];
		int64_t start_target_block_id = target_block_ids[start_index];

		if (!is_swap_out) {
			// Copy from CPU to GPU
			cudaMemcpyAsync(
				k_cache_ptr + start_target_block_id * block_layer_size_in_bytes,
				k_swap_ptr + start_source_block_id * block_layer_size_in_bytes,
				cur_segment_size_in_bytes,
				cudaMemcpyHostToDevice,
				stream
			);
			cudaMemcpyAsync(
				v_cache_ptr + start_target_block_id * block_layer_size_in_bytes,
				v_swap_ptr + start_source_block_id * block_layer_size_in_bytes,
				cur_segment_size_in_bytes,
				cudaMemcpyHostToDevice,
				stream
			);
		} else {
			// Copy from GPU to CPU
			cudaMemcpyAsync(
				k_swap_ptr + start_target_block_id * block_layer_size_in_bytes,
				k_cache_ptr + start_source_block_id * block_layer_size_in_bytes,
				cur_segment_size_in_bytes,
				cudaMemcpyDeviceToHost,
				stream
			);
			cudaMemcpyAsync(
				v_swap_ptr + start_target_block_id * block_layer_size_in_bytes,
				v_cache_ptr + start_source_block_id * block_layer_size_in_bytes,
				cur_segment_size_in_bytes,
				cudaMemcpyDeviceToHost,
				stream
			);
		}

		next_index = end_index;
	}
}
