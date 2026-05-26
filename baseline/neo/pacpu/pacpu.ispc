/* ISPC kernels for paged attention algorithm on CPU */
#include "dtype.h"

#define K_TILE_WIDTH 2
export void qk_product(
  const uniform int cur_layer,
  const uniform int num_blocks,
  const uniform int seq_len,

  const uniform data_t q[], // [NUM_Q_HEADS, HEAD_DIM]
  const uniform data_t k_cache[], // [..., NUM_LAYERS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM]
  const uniform int block_table[], // [seq_len]
  
  uniform itmd_t a[] // [seq_len, NUM_KV_HEADS, HEAD_DIM]
) {
  uniform int imax = seq_len / BLOCK_SIZE + 1;
  for (uniform int i = 0; i < imax; i++) {
    const uniform data_t * k = k_cache + 
      (1ll * cur_layer * num_blocks + block_table[i]) * BLOCK_NELEM;
    for (uniform int j = 0; j < NUM_KV_HEADS; j++) {
      uniform int q_off = j * QH_PER_KVH * HEAD_DIM;
      uniform int k_off = j * BLOCK_SIZE * HEAD_DIM;
      uniform int a_off = i * BLOCK_SIZE * NUM_Q_HEADS + j * QH_PER_KVH;
      uniform int tmax = min(BLOCK_SIZE, seq_len - i * BLOCK_SIZE);
      uniform int t;
      for (t = 0; t < tmax - K_TILE_WIDTH + 1; t += K_TILE_WIDTH) {
        itmd_t sum[QH_PER_KVH][K_TILE_WIDTH];
        for (uniform int h = 0; h < QH_PER_KVH; h++) {
          for (uniform int g = 0; g < K_TILE_WIDTH; g++) {
            sum[h][g] = 0;
          }
        }
        foreach (l = 0 ... HEAD_DIM) {
          for (uniform int g = 0; g < K_TILE_WIDTH; g++) {
            itmd_t k_val = k[k_off + g * HEAD_DIM + l];
            for (uniform int h = 0; h < QH_PER_KVH; h++) {
              sum[h][g] += q[q_off + h * HEAD_DIM + l] * k_val;
            }
          }
        }
        for (uniform int h = 0; h < QH_PER_KVH; h++) {
          for (uniform int g = 0; g < K_TILE_WIDTH; g++) {
            a[a_off + (t + g) * NUM_Q_HEADS + h] = reduce_add(sum[h][g]);
          }
        }
        k_off += HEAD_DIM * K_TILE_WIDTH;
      }
      for (; t < tmax; t++) {
        itmd_t sum[QH_PER_KVH];
        for (uniform int h = 0; h < QH_PER_KVH; h++) {
          sum[h] = 0;
        }
        foreach (l = 0 ... HEAD_DIM) {
          itmd_t k_val = k[k_off + l];
          for (uniform int h = 0; h < QH_PER_KVH; h++) {
            sum[h] += q[q_off + h * HEAD_DIM + l] * k_val;
          }
        }
        for (uniform int h = 0; h < QH_PER_KVH; h++) {
          a[a_off + t * NUM_Q_HEADS + h] = reduce_add(sum[h]);
        }
        k_off += HEAD_DIM;
      }
    }
  }
}

export void av_product(
  const uniform int cur_layer,
  const uniform int num_blocks,
  const uniform int seq_len,

  const uniform itmd_t a[], // [seq_len, NUM_KV_HEADS, HEAD_DIM]
  const uniform data_t v_cache[], // [..., NUM_LAYERS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM]
  const uniform int block_table[], // [seq_len]
  
  uniform otpt_t o[] // [NUM_Q_HEADS, HEAD_DIM]
) {
  uniform int imax = seq_len / BLOCK_SIZE + 1;

  memset(o, 0, NUM_Q_HEADS * HEAD_DIM * sizeof(uniform otpt_t));
  for (uniform int i = 0; i < imax; i++) {
    const uniform data_t * uniform v = v_cache + 
      (1ll * cur_layer * num_blocks + block_table[i]) * BLOCK_NELEM;
    for (uniform int j = 0; j < NUM_KV_HEADS; j++) {
      uniform int o_off = j * QH_PER_KVH * HEAD_DIM;
      uniform int v_off = j * BLOCK_SIZE * HEAD_DIM;
      uniform int tmax = min(BLOCK_SIZE, seq_len - i * BLOCK_SIZE);
      for (uniform int t = 0; t < tmax; t++) {
        uniform int a_off = (i * BLOCK_SIZE + t) * NUM_Q_HEADS + j * QH_PER_KVH;
        foreach (l = 0 ... HEAD_DIM) {
          otpt_t v_val = v[v_off + l];
          for (uniform int h = 0; h < QH_PER_KVH; h++) {
            o[o_off + h * HEAD_DIM + l] += v_val * a[a_off + h];
          }
        }
        v_off += HEAD_DIM;
      }
    }
  }
}

void softmax(
  const uniform int seq_len,
  const uniform itmd_t softmax_scale,

  uniform itmd_t a[], // [seq_len, NUM_Q_HEADS]
  uniform itmd_t asb[] // [NUM_Q_HEADS]
) {
  uniform itmd_t amb[NUM_Q_HEADS];

  foreach (h = 0 ... NUM_Q_HEADS) {
    amb[h] = -1e20;
    for (uniform int i = 0; i < seq_len; i++) {
      a[i * NUM_Q_HEADS + h] *= softmax_scale;
      amb[h] = max(amb[h], a[i * NUM_Q_HEADS + h]);
    }
  }

  foreach (h = 0 ... NUM_Q_HEADS) {
    asb[h] = 0;
    for (uniform int i = 0; i < seq_len; i++) {
      a[i * NUM_Q_HEADS + h] = exp(a[i * NUM_Q_HEADS + h] - amb[h]);
      asb[h] += a[i * NUM_Q_HEADS + h];
    }
  }

  foreach (h = 0 ... NUM_Q_HEADS) {
    for (uniform int i = 0; i < seq_len; i++) {
      a[i * NUM_Q_HEADS + h] /= asb[h];
    }
    asb[h] = log(asb[h]) + amb[h];
  }
}

export void attn_one_seq(
  const uniform int cur_layer,
  const uniform int num_blocks,
  const uniform int seq_len,
  const uniform itmd_t softmax_scale,

  const uniform data_t q[], // [NUM_Q_HEADS, HEAD_DIM]
  const uniform data_t k_cache[], // [..., num_blocks, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM]
  const uniform data_t v_cache[], // [..., num_blocks, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM]
  const uniform int block_table[], // [seq_len]

  uniform itmd_t a[], // [seq_len, NUM_KV_HEADS, HEAD_DIM]
  uniform otpt_t o[], // [NUM_Q_HEADS, HEAD_DIM]
  uniform itmd_t asb[] // [NUM_Q_HEADS]
) {
  qk_product(cur_layer, num_blocks, seq_len, q, k_cache, block_table, a);
  softmax(seq_len, softmax_scale, a, asb);
  av_product(cur_layer, num_blocks, seq_len, a, v_cache, block_table, o);
}

export void gather_output_one_seq(
  const uniform int num_blocks,
  const uniform otpt_t o_buf[], // [num_blocks, NUM_Q_HEADS, HEAD_DIM]

  uniform itmd_t as_buf[], // [num_blocks, NUM_Q_HEADS]
  uniform otpt_t o[] // [NUM_Q_HEADS, HEAD_DIM]
) {
  uniform itmd_t as_all[NUM_Q_HEADS];
  uniform itmd_t am_all[NUM_Q_HEADS];

  foreach(h = 0 ... NUM_Q_HEADS) {
    am_all[h] = -1e20;
    for (uniform int i = 0; i < num_blocks; i++) {
      am_all[h] = max(am_all[h], as_buf[i * NUM_Q_HEADS + h]);
    }
  }

  foreach(h = 0 ... NUM_Q_HEADS) {
    as_all[h] = 0;
    for (uniform int i = 0; i < num_blocks; i++) {
      as_buf[i * NUM_Q_HEADS + h] = exp(as_buf[i * NUM_Q_HEADS + h] - am_all[h]);
      as_all[h] += as_buf[i * NUM_Q_HEADS + h];
    }
  }

  foreach(h = 0 ... NUM_Q_HEADS) {
    for (uniform int i = 0; i < num_blocks; i++) {
      as_buf[i * NUM_Q_HEADS + h] /= as_all[h];
    }
  }

  memset(o, 0, NUM_Q_HEADS * HEAD_DIM * sizeof(uniform otpt_t));
  for (uniform int i = 0; i < num_blocks; i++) {
    uniform int o_off = 0;
    uniform int o_buf_off = i * NUM_Q_HEADS * HEAD_DIM;
    for (uniform int j = 0; j < NUM_Q_HEADS; j++) {
      uniform itmd_t scale = as_buf[i * NUM_Q_HEADS + j];
      foreach (l = 0 ... HEAD_DIM) {
        o[o_off + l] += o_buf[o_buf_off + l] * scale;
      }
      o_off += HEAD_DIM;
      o_buf_off += HEAD_DIM;
    }
  }
}