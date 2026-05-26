#include "attention.h"

template <int NUM_LAYERS, int NUM_Q_HEADS, int NUM_KV_HEADS, int HEAD_DIM, int BLOCK_SIZE>
__global__ void paged_attention_phase1(
  float* __restrict__ mid_o,
  float* __restrict__ mid_o_logexpsum,
  const half* __restrict__ q,
  const half* __restrict__ k,
  const half* __restrict__ v,
  half* __restrict__ kcache,
  half* __restrict__ vcache,
  const float softmax_scale,
  const int* __restrict__ block_table,
  const int* __restrict__ seq_ids,
  const int* __restrict__ seq_lens,
  const int cur_layer,
  const int num_seq_blocks,
  const int seq_block_size,
  const int block_table_width
) {
  // grid shape: [num_decoding_seqs, NUM_Q_HEADS, num_seq_blocks]
  // block shape: [HEAD_DIM]
  const int QH_PER_KVH = NUM_Q_HEADS / NUM_KV_HEADS;
  const int batch_id = blockIdx.x;
  const int qhead_id = blockIdx.y;
  const int seq_block_id = blockIdx.z;
  const int head_offs = threadIdx.x;
  const int kvhead_id = qhead_id / QH_PER_KVH;

  const int seq_id = seq_ids[batch_id];
  const int seq_len = seq_lens[batch_id];

  const int start_token_id = seq_block_id * seq_block_size;

  if (start_token_id >= seq_len) {
    return;
  }

  if (start_token_id + seq_block_size >= seq_len) {
    // The last sequence block, need to store new KV
    // Note that the same value may be stored by different thread blocks, but it's safe since the value is the same
    const int last_block_pos = (seq_len - 1) / BLOCK_SIZE;
    const int last_tok_offs = (seq_len - 1) % BLOCK_SIZE;
    const int last_block_id = block_table[seq_id * block_table_width + last_block_pos];
    const int kv_offs = (batch_id *
      NUM_KV_HEADS + kvhead_id) * 
      HEAD_DIM + head_offs;
    const int64_t kvcache_offs = ((((int64_t)last_block_id * 
      NUM_LAYERS + cur_layer) * 
      NUM_KV_HEADS + kvhead_id) *
      BLOCK_SIZE + last_tok_offs) *
      HEAD_DIM + head_offs;
    kcache[kvcache_offs] = k[kv_offs];
    vcache[kvcache_offs] = v[kv_offs];
  }

  // We group blocks into sections so each thread can handle a "token" in the section
  // to compute Q*K^T
  const int SECTION_SIZE = HEAD_DIM;
  const int sec_offs = head_offs;

  __shared__ float acc[HEAD_DIM];
  __shared__ float cur[HEAD_DIM];
  __shared__ float a[HEAD_DIM]; // A = Q * K^T
  float max_score = -1e20f;
  float sum_exp = 0.0f;
  acc[head_offs] = 0.0f;

  const int maxi = min(seq_len, start_token_id + seq_block_size);
  const int q_offs = (batch_id * 
        NUM_Q_HEADS + qhead_id) * 
        HEAD_DIM;
  for (int i = start_token_id; i < maxi; i += SECTION_SIZE) {
    // Step1: Compute Q * K^T
    if (i + sec_offs < seq_len) {
      const int block_id = block_table[
        seq_id * block_table_width + (i + sec_offs) / BLOCK_SIZE];
      const int block_offs = sec_offs % BLOCK_SIZE;
      const int kcache_offs = ((((int64_t)block_id * 
        NUM_LAYERS + cur_layer) * 
        NUM_KV_HEADS + kvhead_id) * 
        BLOCK_SIZE + block_offs) * 
        HEAD_DIM;
      a[sec_offs] = 0.0f;
      for (int j = 0; j < HEAD_DIM; j++) {
        a[sec_offs] += __half2float(kcache[kcache_offs + j]) * __half2float(q[q_offs + j]);
      }
      a[sec_offs] *= softmax_scale;
    }
    else {
      a[sec_offs] = -1e20f;
    }
    __syncthreads();
    // Step2: Compute softmax (TODO: optimize the reduction)
    __shared__ float cur_max_score;
    __shared__ float new_max_score;
    __shared__ float old_acc_scale;
    __shared__ float cur_sum_exp;
    if (sec_offs == 0) {
      cur_max_score = -1e20f;
      cur_sum_exp = 0.0f;
      for (int j = 0; j < SECTION_SIZE; j++) {
        cur_max_score = fmaxf(cur_max_score, a[j]);
      }
      new_max_score = fmaxf(max_score, cur_max_score);
      old_acc_scale = exp2f(max_score - new_max_score);
      for (int j = 0; j < SECTION_SIZE; j++) {
        a[j] = exp2f(a[j] - new_max_score);
        cur_sum_exp += a[j];
      }
    }
    __syncthreads();
    sum_exp = sum_exp * old_acc_scale + cur_sum_exp;
    max_score = new_max_score;
    // Step3: Compute exp(a - max_score) and o
    cur[head_offs] = 0.0f;
    const int maxj = min(maxi, i + SECTION_SIZE);
    for (int j = i; j < maxj; j += BLOCK_SIZE) {
      const int block_id = block_table[
        seq_id * block_table_width + j / BLOCK_SIZE];
      const int vcache_offs = ((((int64_t)block_id * 
        NUM_LAYERS + cur_layer) * 
        NUM_KV_HEADS + kvhead_id) * 
        BLOCK_SIZE) * 
        HEAD_DIM + head_offs;
      const int maxl = min(maxj, j + BLOCK_SIZE);
      for (int l = j; l < maxl; l++) {
        cur[head_offs] += a[l - i] * __half2float(vcache[vcache_offs + (l - j) * HEAD_DIM]);
      }
    }
    acc[head_offs] = acc[head_offs] * old_acc_scale + cur[head_offs];
  }
  const int mid_o_logexpsum_offs = (batch_id * 
    NUM_Q_HEADS + qhead_id) * 
    num_seq_blocks + seq_block_id;
  const int mid_o_offs = mid_o_logexpsum_offs * 
    HEAD_DIM + head_offs;
  mid_o[mid_o_offs] = acc[head_offs] / sum_exp;
  if (head_offs == 0) {
    mid_o_logexpsum[mid_o_logexpsum_offs] = log2f(sum_exp) + max_score;
  }
}

#define LAUNCH_PHASE1_KERNEL(num_layers, num_q_heads, num_kv_heads, head_dim, block_size) \
  paged_attention_phase1<num_layers, num_q_heads, num_kv_heads, head_dim, block_size> \
    <<<grid1, head_dim>>>( \
      (float*)mid_o.data_ptr(), \
      (float*)mid_o_logexpsum.data_ptr(), \
      (half*)q.data_ptr(), \
      (half*)k.data_ptr(), \
      (half*)v.data_ptr(), \
      (half*)kcache.data_ptr(), \
      (half*)vcache.data_ptr(), \
      softmax_scale, \
      (int*)block_table.data_ptr(), \
      (int*)seq_ids.data_ptr(), \
      (int*)seq_lens.data_ptr(), \
      cur_layer, \
      num_seq_blocks, \
      seq_block_size, \
      block_table_width \
    )

template <int NUM_LAYERS, int NUM_Q_HEADS, int NUM_KV_HEADS, int HEAD_DIM, int BLOCK_SIZE>
__global__ void paged_attention_phase2(
  const float* __restrict__ mid_o,
  const float* __restrict__ mid_o_logexpsum,
  half* __restrict__ o,

  const int* __restrict__ seq_ids,
  const int* __restrict__ seq_lens,

  const int seq_block_size
) {
  // grid shape: [num_decoding_seqs, NUM_Q_HEADS]
  // block shape: [HEAD_DIM]
  const int batch_id = blockIdx.x;
  const int qhead_id = blockIdx.y;
  const int head_offs = threadIdx.x;

  const int seq_len = seq_lens[batch_id];
  const int num_seq_blocks = (seq_len - 1) / seq_block_size + 1;
  float sum_exp = 0.0f;
  float max_score = -1e20f;
  __shared__ float acc[HEAD_DIM];
  acc[head_offs] = 0.0f;

  for (int i = 0; i < num_seq_blocks; i++) {
    const int mid_o_logexpsum_offs = (batch_id * 
      NUM_Q_HEADS + qhead_id) * 
      num_seq_blocks + i;
    const int mid_o_offs = mid_o_logexpsum_offs * 
      HEAD_DIM + head_offs;
    const float cur_mid_o = mid_o[mid_o_offs];
    const float cur_mid_o_logexpsum = mid_o_logexpsum[mid_o_logexpsum_offs];
    const float new_max_score = fmaxf(max_score, cur_mid_o_logexpsum);
    const float old_acc_scale = exp2f(max_score - new_max_score);
    const float exp_score = exp2f(cur_mid_o_logexpsum - new_max_score);
    acc[head_offs] = acc[head_offs] * old_acc_scale + cur_mid_o * exp_score;
    sum_exp = sum_exp * old_acc_scale + exp_score;
    max_score = new_max_score;
  }

  const int o_offs = (batch_id * 
    NUM_Q_HEADS + qhead_id) * 
    HEAD_DIM + head_offs;
  
  o[o_offs] = __float2half(acc[head_offs] / sum_exp);
}

#define LAUNCH_PHASE2_KERNEL(num_layers, num_q_heads, num_kv_heads, head_dim, block_size) \
  paged_attention_phase2<num_layers, num_q_heads, num_kv_heads, head_dim, block_size> \
    <<<grid2, head_dim>>>( \
      (float*)mid_o.data_ptr(), \
      (float*)mid_o_logexpsum.data_ptr(), \
      (half*)o.data_ptr(), \
      (int*)seq_ids.data_ptr(), \
      (int*)seq_lens.data_ptr(), \
      seq_block_size \
    )

#define SELECT_KERNEL(block_size) \
  if (num_layers == 32 && num_q_heads == 32 && num_kv_heads == 8 && head_dim == 128) { \
    LAUNCH_PHASE1_KERNEL(32, 32, 8, 128, block_size); \
    LAUNCH_PHASE2_KERNEL(32, 32, 8, 128, block_size); \
  } \
  else if (num_layers == 32 && num_q_heads == 32 && num_kv_heads == 32 && head_dim == 128) { \
    LAUNCH_PHASE1_KERNEL(32, 32, 32, 128, block_size); \
    LAUNCH_PHASE2_KERNEL(32, 32, 32, 128, block_size); \
  } \
  else if (num_layers == 40 && num_q_heads == 40 && num_kv_heads == 40 && head_dim == 128) { \
    LAUNCH_PHASE1_KERNEL(40, 40, 40, 128, block_size); \
    LAUNCH_PHASE2_KERNEL(40, 40, 40, 128, block_size); \
  } \
  else if (num_layers == 80 && num_q_heads == 64 && num_kv_heads == 8 && head_dim == 128) { \
    LAUNCH_PHASE1_KERNEL(80, 64, 8, 128, block_size); \
    LAUNCH_PHASE2_KERNEL(80, 64, 8, 128, block_size); \
  } \
  else { \
    throw std::runtime_error("Unsupported configuration"); \
  }

void paged_attention(
  torch::Tensor q, // [num_decoding_seqs, num_q_heads, head_dim]
  torch::Tensor k, // [num_decoding_seqs, num_kv_heads, head_dim]
  torch::Tensor v, // [num_decoding_seqs, num_kv_heads, head_dim]
  torch::Tensor o, // [num_decoding_seqs, num_q_heads, head_dim]
  torch::Tensor kcache, // [..., num_layers, num_kv_heads, block_size, head_dim]
  torch::Tensor vcache, // [..., num_layers, num_kv_heads, block_size, head_dim]
  float softmax_scale,
  torch::Tensor block_table, // [..., block_table_width]
  torch::Tensor seq_ids,   // [num_decoding_seqs]
  torch::Tensor seq_lens,  // [num_decoding_seqs]
  const int64_t cur_layer,
  const int64_t seq_block_size,
  const int64_t num_seq_blocks
) {
  const int num_decoding_seqs = q.size(0);
  const int num_q_heads = q.size(1);
  const int head_dim = q.size(2);
  const int num_kv_heads = k.size(1);
  const int num_layers = kcache.size(1);
  const int block_size = kcache.size(-2);
  const int block_table_width = block_table.size(-1);

  assert (seq_block_size % block_size == 0);
  assert (seq_block_size % head_dim == 0);

  auto options = torch::TensorOptions()
    .dtype(torch::kFloat32)
    .device(torch::kCUDA);

  torch::Tensor mid_o = torch::empty(
    {num_decoding_seqs, num_q_heads, num_seq_blocks, head_dim},
    options
  );
  torch::Tensor mid_o_logexpsum = torch::zeros(
    {num_decoding_seqs, num_q_heads, num_seq_blocks},
    options
  );

  // softmax is defined to use exp, here we use 2^x so we need to scale the exponent by log2(e)
  softmax_scale *= 1.442695040888963;

  dim3 grid1(num_decoding_seqs, num_q_heads, num_seq_blocks);
  dim3 grid2(num_decoding_seqs, num_q_heads);

  switch(block_size){
    case 16:
      SELECT_KERNEL(16);
      break;
    case 32:
      SELECT_KERNEL(32);
      break;
    case 64:
      SELECT_KERNEL(64);
      break;
    default:
      throw std::runtime_error("Unsupported block size");
  }
}