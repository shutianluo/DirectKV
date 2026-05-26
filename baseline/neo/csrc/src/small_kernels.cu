/*
 * Code adapted from https://github.com/LLMServe/SwiftTransformer/tree/main
 */

#include "small_kernels.h"

template<typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffffu, val, mask, 32);
    return val;
}

/* Calculate the sum of all elements in a block */
template<typename T>
__inline__ __device__ T blockReduceSum(T val) {
    static __shared__ T shared[32];
    int64_t lane = threadIdx.x & 0x1fu;
    int64_t wid = threadIdx.x >> 5;

    val = warpReduceSum<T>(val);

    if (lane == 0)
        shared[wid] = val;

    __syncthreads();

    // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
    // blockDim.x is not divided by 32
    val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : (T)(0.0f);
    val = warpReduceSum<T>(val);
    return val;
}


__global__ void fused_add_rmsnorm_inplace_Kernel(
    half* __restrict__ buffer,          // [num_tokens, hidden_size]
    half* __restrict__ residual,        // [num_tokens, hidden_size]
    const half* __restrict__ weight,    // [hidden_size]
    const float epsilon,
    const int64_t hidden_size
) {
    // grid size: num_tokens
    // block size: min(hidden_size, 1024)
    
    // Step 0. Let sum residual with buffer
    for (int64_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        residual[blockIdx.x * hidden_size + i] = __hadd(residual[blockIdx.x * hidden_size + i], buffer[blockIdx.x * hidden_size + i]);
    }
    // Step 1. Every thread computes some part of the sum of squares
    float square_sum = 0.0;
    __shared__ float inv_rms;
    for (int64_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        const float x = __half2float(residual[blockIdx.x * hidden_size + i]);
        square_sum += x * x;
    }
    // Step 2. Sum the squares across threads
    square_sum = blockReduceSum(square_sum);
    // Step 3. Compute the inverse root mean square
    if (threadIdx.x == 0) {
        inv_rms = rsqrtf(square_sum / hidden_size + epsilon);
    }
    __syncthreads();
    // Step 4. Compute the output
    for (int64_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        const float x = __half2float(residual[blockIdx.x * hidden_size + i]);
        const float w = __half2float(weight[i]);
        buffer[blockIdx.x * hidden_size + i] = __float2half(x * w * inv_rms);
    }
}


void fused_add_rmsnorm_inplace(
    torch::Tensor buffer,
    torch::Tensor residual,
    torch::Tensor weight,
    const float epsilon
) {
    // residual += buffer
    // buffer = residual * weight * rsqrt(sum(residual^2) / hidden_size + epsilon)
    const int64_t num_tokens = buffer.size(0);
    const int64_t hidden_size = buffer.size(1);

    const int64_t block_size = std::min(hidden_size, 512L);
    const int64_t grid_size = num_tokens;
    half* buffer_p = (half*)buffer.data_ptr();
    half* residual_p = (half*)residual.data_ptr();
    const half* weight_p = (half*)weight.data_ptr();
    fused_add_rmsnorm_inplace_Kernel<<<grid_size, block_size>>>(buffer_p, residual_p, weight_p, epsilon, hidden_size);
}


__global__ void silu_and_mul_inplace_Kernel(
    half* __restrict__ buffer,                // [num_tokens, hidden_size * 2]
    const int64_t hidden_size
) {
    // grid size: num_tokens
    // block size: hidden_size
    int64_t off = blockIdx.x * hidden_size * 2;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float x = __half2float(buffer[off + i + hidden_size]);
        buffer[off + i] = __hmul(__float2half(x / (1 + __expf(-x))), buffer[off + i]);
    }
}


void silu_and_mul_inplace(
    torch::Tensor buffer
) {
    // buffer[:, :hidden] = silu(buffer[:, hidden:]) * buffer[:, :hidden]
    const int64_t num_tokens = buffer.size(0);
    const int64_t hidden_size = buffer.size(1) / 2;

    const int64_t grid_dim = num_tokens;
    const int64_t block_dim = min(hidden_size, 512L);
    
    half* buffer_p = (half*)buffer.data_ptr();
    silu_and_mul_inplace_Kernel<<<grid_dim, block_dim>>>(buffer_p, hidden_size);
}


template<int NUM_Q_HEADS, int NUM_K_HEADS>
__global__ void rotary_embedding_inplace_Kernel(
    half* __restrict__ q,            // [num_tokens, num_q_heads, head_dim]
    half* __restrict__ k,            // [num_tokens, num_k_heads, head_dim]
    const half* __restrict__ sin_table,   // [num_tokens, head_dim / 2]
    const half* __restrict__ cos_table    // [num_tokens, head_dim / 2]
) {
    const int64_t token_idx = blockIdx.x;
    const int64_t k_head_idx = blockIdx.y;
    const int64_t dim_idx = threadIdx.x;
    const int64_t head_dim = blockDim.x * 2;
    const int64_t QH_PER_KVH = NUM_Q_HEADS / NUM_K_HEADS;

    const int64_t t_off = token_idx * blockDim.x + dim_idx;
    const half sin_val = sin_table[t_off];
    const half cos_val = cos_table[t_off];
    const int64_t k_off0 = (token_idx * NUM_K_HEADS + k_head_idx) * head_dim + dim_idx;
    const int64_t k_off1 = k_off0 + blockDim.x;
    const half k0 = k[k_off0];
    const half k1 = k[k_off1];
    k[k_off0] = __hsub(__hmul(k0, cos_val), __hmul(k1, sin_val));
    k[k_off1] = __hadd(__hmul(k0, sin_val), __hmul(k1, cos_val));

    #pragma unroll
    for (int i = 0; i < QH_PER_KVH; i++) {
        const int64_t q_head_idx = k_head_idx * QH_PER_KVH + i;
        const int64_t q_off0 = (token_idx * NUM_Q_HEADS + q_head_idx) * head_dim + dim_idx;
        const int64_t q_off1 = q_off0 + blockDim.x;
        const half q0 = q[q_off0];
        const half q1 = q[q_off1];
        q[q_off0] = __hsub(__hmul(q0, cos_val), __hmul(q1, sin_val));
        q[q_off1] = __hadd(__hmul(q0, sin_val), __hmul(q1, cos_val));
    }
}


#define ROTARY_EMBEDDING_INPLACE_KERNEL(qh, kh) \
    rotary_embedding_inplace_Kernel<qh, kh><<<grid, block_size>>>(q_p, k_p, sin_table_p, cos_table_p);
#define TRY_ROTARY_EMBEDDING_INPLACE_KERNEL(qh, kh) \
    if (num_q_heads == qh && num_k_heads == kh) { \
        ROTARY_EMBEDDING_INPLACE_KERNEL(qh, kh); \
        return; \
    }

void rotary_embedding_inplace(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor sin_table,
    torch::Tensor cos_table
) {
    const int64_t num_tokens = q.size(0);
    const int64_t num_q_heads = q.size(1);
    const int64_t num_k_heads = k.size(1);
    const int64_t head_dim = q.size(2);
    const int64_t qh_per_kvh = num_q_heads / num_k_heads;

    const dim3 grid(num_tokens, num_k_heads);
    const int64_t block_size = head_dim / 2;

    half* q_p = (half*)q.data_ptr();
    half* k_p = (half*)k.data_ptr();
    const half* sin_table_p = (half*)sin_table.data_ptr();
    const half* cos_table_p = (half*)cos_table.data_ptr();

    if (head_dim != 128) {
        throw std::invalid_argument("head_dim must be 128");
    }

    TRY_ROTARY_EMBEDDING_INPLACE_KERNEL(32, 8);  // Llama-3-8B
    TRY_ROTARY_EMBEDDING_INPLACE_KERNEL(16, 16); // Llama-2-7B TP=2
    TRY_ROTARY_EMBEDDING_INPLACE_KERNEL(32, 32); // Llama-2-7B
    TRY_ROTARY_EMBEDDING_INPLACE_KERNEL(40, 40); // Llama-2-13B
    TRY_ROTARY_EMBEDDING_INPLACE_KERNEL(8, 1);   // Llama-2/3-70B TP=8
    TRY_ROTARY_EMBEDDING_INPLACE_KERNEL(16, 2);  // Llama-2/3-70B TP=4
    TRY_ROTARY_EMBEDDING_INPLACE_KERNEL(32, 4);  // Llama-2/3-70B TP=2
    TRY_ROTARY_EMBEDDING_INPLACE_KERNEL(64, 8);  // Llama-2/3-70B TP=1
    auto error_string = "Unsupported num_q_heads and num_k_heads: " + std::to_string(num_q_heads) + ", " + std::to_string(num_k_heads);
    throw std::invalid_argument(error_string);
}


template<int NUM_KV_HEADS>
__global__ void prefill_store_kvcache_Kernel(
    const half* __restrict__ k, // [num_tokens, num_heads, head_dim]
    const half* __restrict__ v, // [num_tokens, num_heads, head_dim]
    half* __restrict__ k_cache, // [... , num_blocks, num_kv_heads, block_size, head_dim]
    half* __restrict__ v_cache, // [... , num_blocks, num_kv_heads, block_size, head_dim]
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_ids,
    const int32_t* __restrict__ seq_start_locs,
    const int32_t* __restrict__ seq_lens,
    const int64_t itm_layer,
    const int64_t gpu_layer,
    const int64_t num_cprfs,
    const int64_t block_size,
    const int64_t block_table_width,
    const int64_t num_blocks,
    const int64_t head_dim
) {
    // grid size: [num_seqs, (max_pref_len - 1) / block_size + 1]
    // block size: head_dim
    const int64_t batch_pos = blockIdx.x;
    const int64_t block_pos = blockIdx.y;
    const int64_t seq_len = seq_lens[batch_pos];
    if (block_size * block_pos >= seq_len) {
        return;
    }

    const int64_t cur_layer = batch_pos < num_cprfs ? itm_layer : gpu_layer;
    const int32_t seq_id = seq_ids[batch_pos];
    const int32_t seq_start_loc = seq_start_locs[batch_pos];
    const int32_t block_idx = block_table[seq_id * block_table_width + block_pos];
    const int32_t max_i = min(block_size, seq_len - block_size * block_pos); 

    for (int i = 0; i < max_i; i++) {
        int64_t src_off = 
            (seq_start_loc + block_pos * block_size + i) * NUM_KV_HEADS * head_dim + threadIdx.x;
        int64_t dst_off =
            ((cur_layer * num_blocks + block_idx) * NUM_KV_HEADS * block_size + i) * head_dim + threadIdx.x;
        #pragma unroll
        for (int j = 0; j < NUM_KV_HEADS; j++) {
            k_cache[dst_off] = k[src_off];
            v_cache[dst_off] = v[src_off];
            src_off += head_dim;
            dst_off += block_size * head_dim;
        }
    }
}


#define PREFILL_STORE_KVCACHE_KERNEL(kvh) \
    prefill_store_kvcache_Kernel<kvh><<<grid, block_dim_x>>>( \
        k_p, v_p, k_cache_p, v_cache_p, \
        block_table_p, seq_ids_p, seq_start_locs_p, seq_lens_p, \
        itm_layer, gpu_layer, num_cprfs, \
        block_size, block_table_width, num_blocks, head_dim \
    );
#define TRY_PREFILL_STORE_KVCACHE_KERNEL(kvh) \
    if (num_kv_heads == kvh) { \
        PREFILL_STORE_KVCACHE_KERNEL(kvh); \
        return; \
    }


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
) {
    const int64_t num_blocks = k_cache.size(1);
    const int64_t num_kv_heads = k_cache.size(2);
    const int64_t block_size = k_cache.size(3);
    const int64_t head_dim = k_cache.size(4);
    const int64_t block_table_width = block_table.size(1);
    const int64_t num_seqs = seq_ids.size(0);

    const dim3 grid(num_seqs, (max_pref_len - 1) / block_size + 1);
    const int64_t block_dim_x = head_dim;

    half* k_p = (half*)k.data_ptr();
    half* v_p = (half*)v.data_ptr();
    half* k_cache_p = (half*)k_cache.data_ptr();
    half* v_cache_p = (half*)v_cache.data_ptr();
    const int32_t* block_table_p = (int32_t*)block_table.data_ptr();
    const int32_t* seq_ids_p = (int32_t*)seq_ids.data_ptr();
    const int32_t* seq_start_locs_p = (int32_t*)seq_start_locs.data_ptr();
    const int32_t* seq_lens_p = (int32_t*)seq_lens.data_ptr();

    if (head_dim != 128) {
        throw std::invalid_argument("head_dim must be 128");
    }

    TRY_PREFILL_STORE_KVCACHE_KERNEL(1);  // Llama-2/3-70B TP=8
    TRY_PREFILL_STORE_KVCACHE_KERNEL(2);  // Llama-2/3-70B TP=4
    TRY_PREFILL_STORE_KVCACHE_KERNEL(4);  // Llama-2/3-70B TP=2
    TRY_PREFILL_STORE_KVCACHE_KERNEL(8);  // Llama-2/3-70B TP=1, Llama-3-8B
    TRY_PREFILL_STORE_KVCACHE_KERNEL(16); // Llama-2-7B TP=2
    TRY_PREFILL_STORE_KVCACHE_KERNEL(32); // Llama-2-7B TP=1
    TRY_PREFILL_STORE_KVCACHE_KERNEL(40); // Llama-2-13B
    TRY_PREFILL_STORE_KVCACHE_KERNEL(56); // OPT-30B
    auto error_string = "Unsupported num_kv_heads: " + std::to_string(num_kv_heads);
    throw std::invalid_argument(error_string);
}


__global__ void embedding_kernel(
    const int32_t* __restrict__ input_tokens, // [batch_size]
    const half* __restrict__ weights,         // [vocab_size, hidden_size]
    half* __restrict__ output,                // [batch_size, hidden_size]
    const int64_t hidden_size,
    const int64_t vocab_size,
    const int64_t token_offset
) {
    // grid size: batch_size
    // block size: some factor of hidden_size (i.e. 128)
    const int64_t batch_pos = blockIdx.x;
    const int64_t hidden_pos = threadIdx.x;
    const int64_t input_token = input_tokens[batch_pos] - token_offset;
    if (input_token < 0 || input_token >= vocab_size) {
        return;
    }
    const int64_t weight_off = input_token * hidden_size + hidden_pos;
    const int64_t out_off = batch_pos * hidden_size + hidden_pos;
    for (int i = 0; i < hidden_size; i += blockDim.x) {
        output[out_off + i] = weights[weight_off + i];
    }
}


void embedding(
    torch::Tensor input_tokens, // [batch_size]
    torch::Tensor weights,      // [vocab_size, hidden_size]
    torch::Tensor output,       // [batch_size, hidden_size]
    const int64_t token_offset
) {
    // weights is a slice of the input_tokens embedding matrix, we set the embedding
    // in output if the token is within the range, otherwise we leave it as zeros
    const int64_t batch_size = input_tokens.size(0);
    const int64_t hidden_size = weights.size(1);
    const int64_t vocab_size = weights.size(0);

    int block_size = 128;
    assert(hidden_size % block_size == 0);

    embedding_kernel<<<batch_size, block_size>>>(
        (int32_t*)input_tokens.data_ptr(),
        (half*)weights.data_ptr(),
        (half*)output.data_ptr(),
        hidden_size,
        vocab_size,
        token_offset
    );
}