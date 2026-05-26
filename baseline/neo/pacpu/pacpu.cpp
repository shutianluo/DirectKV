#include <torch/library.h>
#include <ATen/Tensor.h>

#include "dtype.h"
#include "core.h"

typedef at::Half at_data_t;

// #define USE_ATEN_OPER
#ifdef USE_ATEN_OPER
#include <ATen/TensorIndexing.h>
#include <ATen/TensorOperators.h>

void paged_attention_cpu_torch(
  int64_t cur_layer,
  double softmax_scale,
	const std::vector<int64_t> &decoding_seq_ids,
	const std::vector<int64_t> &decoding_seq_lengths,

  at::Tensor q,
  at::Tensor k,
  at::Tensor v,
	at::Tensor k_cache,
	at::Tensor v_cache,
  at::Tensor block_table,
  at::Tensor o
) {
  for (auto i = 0; i < decoding_seq_ids.size(); i++) {
    auto seq_id = decoding_seq_ids[i];
    auto seq_len = decoding_seq_lengths[i];
    at::Tensor qi = q.index({i});
    at::Tensor ki = k.index({i});
    at::Tensor vi = v.index({i});
    auto blkid = (seq_len - 1) / BLOCK_SIZE;
    auto blkoff = (seq_len - 1) % BLOCK_SIZE;
    std::cout << k_cache.sizes() << std::endl;
    k_cache.index_put_({cur_layer, blkid, at::indexing::Slice(), blkoff, at::indexing::Slice()}, ki);
    v_cache.index_put_({cur_layer, blkid, at::indexing::Slice(), blkoff, at::indexing::Slice()}, vi);
    at::Tensor nk = k_cache.index({cur_layer, block_ids}).permute({1, 2, 0}).to(at::kFloat);
    at::Tensor nv = v_cache.index({cur_layer, block_ids}).permute({1, 0, 2}).to(at::kFloat);
    at::Tensor attn_score = (at::bmm(qi, nk) * softmax_scale).softmax(-1);
    o.index_put_({i}, at::bmm(attn_score, nv).view(-1));
  }
}
#endif

void assert_hyper_params_expected(int num_q_heads, int num_kv_heads, int num_layers, int head_dim, int block_size) {
  if (num_q_heads != NUM_Q_HEADS) {
    throw std::invalid_argument("expected num_q_heads to be " + std::to_string(NUM_Q_HEADS) + ", but got " + std::to_string(num_q_heads));
  }
  if (num_kv_heads != NUM_KV_HEADS) {
    throw std::invalid_argument("expected num_kv_heads to be " + std::to_string(NUM_KV_HEADS) + ", but got " + std::to_string(num_kv_heads));
  }
  if (num_layers != NUM_LAYERS) {
    throw std::invalid_argument("expected num_layers to be " + std::to_string(NUM_LAYERS) + ", but got " + std::to_string(num_layers));
  }
  if (head_dim != HEAD_DIM) {
    throw std::invalid_argument("expected head_dim to be " + std::to_string(HEAD_DIM) + ", but got " + std::to_string(head_dim));
  }
  if (block_size != BLOCK_SIZE) {
    throw std::invalid_argument("expected block_size to be " + std::to_string(BLOCK_SIZE) + ", but got " + std::to_string(block_size));
  }
}

/*
 * Paged attention backend selection:
 *   USE_ISPC_TASKS_OPER  — ISPC + OpenMP tasks (original, x86 only)
 *   USE_ISPC_OPER        — ISPC without tasks (x86 only)
 *   USE_OMP_OPER         — C++ + OpenMP SIMD (portable; auto-selected on aarch64)
 *   USE_BRUTE_OPER       — scalar fallback (correctness reference)
 */
#if defined(__aarch64__)
#  define USE_OMP_OPER
#else
#  define USE_ISPC_TASKS_OPER
// #define USE_ISPC_OPER
#endif

void paged_attention_cpu(
  int64_t cur_layer,
  double softmax_scale,
	const std::vector<int64_t> &seq_ids,
	const std::vector<int64_t> &seq_lengths,

  at::Tensor q, // [batch_size, num_q_heads, head_dim]
  at::Tensor k, // [batch_size, num_kv_heads, head_dim]
  at::Tensor v, // [batch_size, num_kv_heads, head_dim]
	at::Tensor k_cache, // [..., num_layers, num_kv_heads, block_size, head_dim]
	at::Tensor v_cache, // [..., num_layers, num_kv_heads, block_size, head_dim]
  at::Tensor block_table, // [..., max_seq_len]
  at::Tensor o // [batch_size, num_kv_heads * qh_per_kvh * head_dim]
) {
  int batch_size = q.size(0);
  int num_q_heads = q.size(1);
  int num_layers = k_cache.size(0);
  int num_blocks = k_cache.size(1);
  int num_kv_heads = k_cache.size(2);
  int block_size = k_cache.size(3);
  int head_dim = k_cache.size(4);
  int block_table_width = block_table.size(1);

  assert_hyper_params_expected(num_q_heads, num_kv_heads, num_layers, head_dim, block_size);

  auto qbatch_p = (data_t*) q.data_ptr<at_data_t>();
  auto kbatch_p = (data_t*) k.data_ptr<at_data_t>();
  auto vbatch_p = (data_t*) v.data_ptr<at_data_t>();
  auto obatch_p = o.data_ptr<otpt_t>();
  auto kcache_p = (data_t*) k_cache.data_ptr<at_data_t>();
  auto vcache_p = (data_t*) v_cache.data_ptr<at_data_t>();
  auto block_table_p = block_table.data_ptr<int32_t>(); // [batch_size, max_seq_len]

  #ifdef USE_BRUTE_OPER
    brute_attention(
      cur_layer, num_blocks, batch_size, block_table_width, softmax_scale,
      seq_ids, seq_lengths,
      qbatch_p, kbatch_p, vbatch_p, obatch_p, kcache_p, vcache_p, block_table_p
    );
  #elif defined(USE_ISPC_OPER)
    ispc_attention(
      cur_layer, num_blocks, batch_size, block_table_width, softmax_scale,
      seq_ids, seq_lengths,
      qbatch_p, kbatch_p, vbatch_p, obatch_p, kcache_p, vcache_p, block_table_p
    );
  #elif defined(USE_ISPC_TASKS_OPER)
    ispc_attention_tasks(
      cur_layer, num_blocks, batch_size, block_table_width, softmax_scale,
      seq_ids, seq_lengths,
      qbatch_p, kbatch_p, vbatch_p, obatch_p, kcache_p, vcache_p, block_table_p
    );
  #elif defined(USE_OMP_OPER)
    // ARM/portable path: ispc_attention_tasks re-uses the same ispc:: namespace
    // functions, now backed by pacpu_arm.h instead of the ISPC-generated header.
    ispc_attention_tasks(
      cur_layer, num_blocks, batch_size, block_table_width, softmax_scale,
      seq_ids, seq_lengths,
      qbatch_p, kbatch_p, vbatch_p, obatch_p, kcache_p, vcache_p, block_table_p
    );
  #endif
}

TORCH_LIBRARY(pacpu, m) {
#ifdef USE_ATEN_OPER
  m.def("paged_attention_cpu_torch", &paged_attention_cpu_torch);
#endif
  m.def("paged_attention_cpu", &paged_attention_cpu);
}