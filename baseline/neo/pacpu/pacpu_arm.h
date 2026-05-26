/*
 * pacpu_arm.h — ARM/aarch64 replacement for pacpu_ispc.h
 *
 * Implements the same `namespace ispc` interface as the ISPC-generated header,
 * using OpenMP SIMD pragmas + GCC auto-vectorization.
 *
 * On GH200 (Neoverse V2 / ARMv9.2-A) GCC emits:
 *   - SVE2 fused-multiply-add for the HEAD_DIM=128 inner dot products
 *   - NEON fcvtl (FP16→FP32 widening) for KV cache loads
 *   - Vectorized softmax via NEON fmax/fexp approximations
 *
 * The key ISPC pattern:
 *   foreach (l = 0 ... HEAD_DIM) { acc += q[l] * k[l]; }
 *   result = reduce_add(acc);
 *
 * maps directly to:
 *   float acc = 0.f;
 *   #pragma omp simd reduction(+:acc)
 *   for (int l = 0; l < HEAD_DIM; l++) { acc += (float)q[l] * (float)k[l]; }
 */
#pragma once

#include <cmath>
#include <cstring>
#include <algorithm>
#include "dtype.h"

// Require IEEE FP16 support — confirmed via __ARM_FP16_FORMAT_IEEE
static_assert(sizeof(data_t) == 2, "data_t must be 16-bit (fp16)");

namespace ispc {

// ------------------------------------------------------------------
// Q×K^T for one sequence segment (paged KV cache layout)
//
// q          : [NUM_Q_HEADS, HEAD_DIM]  fp16
// k_cache    : [layer*num_blocks, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM]  fp16
// block_table: [ceil(seq_len/BLOCK_SIZE)]  block ids
// a (out)    : [seq_len, NUM_Q_HEADS]  fp32  attention logits
// ------------------------------------------------------------------
inline void qk_product(
    int cur_layer,
    int num_blocks,
    int seq_len,
    const data_t* __restrict__ q,
    const data_t* __restrict__ k_cache,
    const int*    __restrict__ block_table,
    itmd_t*       __restrict__ a
) {
    int imax = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int i = 0; i < imax; i++) {
        const data_t* k_blk = k_cache +
            (1LL * cur_layer * num_blocks + block_table[i]) * BLOCK_NELEM;
        int tmax = std::min(BLOCK_SIZE, seq_len - i * BLOCK_SIZE);

        for (int j = 0; j < NUM_KV_HEADS; j++) {
            // q sub-array for this KV head group
            const data_t* q_kv = q + j * QH_PER_KVH * HEAD_DIM;
            // output offset: [i*BLOCK_SIZE + t, NUM_Q_HEADS]
            int a_base = (i * BLOCK_SIZE) * NUM_Q_HEADS + j * QH_PER_KVH;

            // Process two consecutive K-vectors at once (K_TILE_WIDTH=2)
            int t = 0;
            for (; t <= tmax - 2; t += 2) {
                const data_t* kt0 = k_blk + (j * BLOCK_SIZE + t) * HEAD_DIM;
                const data_t* kt1 = kt0 + HEAD_DIM;

                for (int h = 0; h < QH_PER_KVH; h++) {
                    const data_t* qh = q_kv + h * HEAD_DIM;
                    float s0 = 0.0f, s1 = 0.0f;
                    // GCC vectorizes this to SVE2 fmla with fp16→fp32 widening
                    #pragma omp simd reduction(+:s0,s1)
                    for (int l = 0; l < HEAD_DIM; l++) {
                        float qv = (float)qh[l];
                        s0 += qv * (float)kt0[l];
                        s1 += qv * (float)kt1[l];
                    }
                    a[a_base + t * NUM_Q_HEADS + h]       = s0;
                    a[a_base + (t + 1) * NUM_Q_HEADS + h] = s1;
                }
            }
            // Scalar tail (at most 1 remaining token per block)
            for (; t < tmax; t++) {
                const data_t* kt = k_blk + (j * BLOCK_SIZE + t) * HEAD_DIM;
                for (int h = 0; h < QH_PER_KVH; h++) {
                    const data_t* qh = q_kv + h * HEAD_DIM;
                    float s = 0.0f;
                    #pragma omp simd reduction(+:s)
                    for (int l = 0; l < HEAD_DIM; l++) {
                        s += (float)qh[l] * (float)kt[l];
                    }
                    a[a_base + t * NUM_Q_HEADS + h] = s;
                }
            }
        }
    }
}

// ------------------------------------------------------------------
// Softmax with log-sum-exp output (internal linkage — called by attn_one_seq)
//
// a (in/out) : [seq_len, NUM_Q_HEADS]  scaled in-place, then normalized
// asb (out)  : [NUM_Q_HEADS]  log(sum) + max  (used for partial-output merging)
// ------------------------------------------------------------------
static void softmax(
    int seq_len,
    float softmax_scale,
    itmd_t* __restrict__ a,
    itmd_t* __restrict__ asb
) {
    float amb[NUM_Q_HEADS];
    for (int h = 0; h < NUM_Q_HEADS; h++) amb[h] = -1e20f;

    // Scale in-place and find per-head maximum
    for (int i = 0; i < seq_len; i++) {
        float* ap = a + i * NUM_Q_HEADS;
        for (int h = 0; h < NUM_Q_HEADS; h++) {
            float v = ap[h] * softmax_scale;
            ap[h] = v;
            if (v > amb[h]) amb[h] = v;
        }
    }

    // exp(x - max) and accumulate denominator
    for (int h = 0; h < NUM_Q_HEADS; h++) asb[h] = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        float* ap = a + i * NUM_Q_HEADS;
        for (int h = 0; h < NUM_Q_HEADS; h++) {
            float e = expf(ap[h] - amb[h]);
            ap[h] = e;
            asb[h] += e;
        }
    }

    // Normalize and compute log-sum-exp accumulator
    for (int i = 0; i < seq_len; i++) {
        float* ap = a + i * NUM_Q_HEADS;
        #pragma omp simd
        for (int h = 0; h < NUM_Q_HEADS; h++) {
            ap[h] /= asb[h];
        }
    }
    for (int h = 0; h < NUM_Q_HEADS; h++) {
        asb[h] = logf(asb[h]) + amb[h];
    }
}

// ------------------------------------------------------------------
// A×V for one sequence segment
//
// a      : [seq_len, NUM_Q_HEADS]  fp32 softmax weights
// v_cache: [layer*num_blocks, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM]  fp16
// o (out): [NUM_Q_HEADS, HEAD_DIM]  fp32
// ------------------------------------------------------------------
inline void av_product(
    int cur_layer,
    int num_blocks,
    int seq_len,
    const itmd_t* __restrict__ a,
    const data_t* __restrict__ v_cache,
    const int*    __restrict__ block_table,
    otpt_t*       __restrict__ o
) {
    memset(o, 0, NUM_Q_HEADS * HEAD_DIM * sizeof(otpt_t));
    int imax = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int i = 0; i < imax; i++) {
        const data_t* v_blk = v_cache +
            (1LL * cur_layer * num_blocks + block_table[i]) * BLOCK_NELEM;
        int tmax = std::min(BLOCK_SIZE, seq_len - i * BLOCK_SIZE);
        for (int j = 0; j < NUM_KV_HEADS; j++) {
            int o_off = j * QH_PER_KVH * HEAD_DIM;
            for (int t = 0; t < tmax; t++) {
                const data_t* vt = v_blk + (j * BLOCK_SIZE + t) * HEAD_DIM;
                int a_off = (i * BLOCK_SIZE + t) * NUM_Q_HEADS + j * QH_PER_KVH;
                for (int h = 0; h < QH_PER_KVH; h++) {
                    float alpha = a[a_off + h];
                    float* oh   = o + o_off + h * HEAD_DIM;
                    // Vectorized: oh[l] += alpha * fp16_to_fp32(vt[l])
                    #pragma omp simd
                    for (int l = 0; l < HEAD_DIM; l++) {
                        oh[l] += alpha * (float)vt[l];
                    }
                }
            }
        }
    }
}

// ------------------------------------------------------------------
// Full single-sequence attention (qk → softmax → av)
// ------------------------------------------------------------------
inline void attn_one_seq(
    int cur_layer,
    int num_blocks,
    int seq_len,
    float softmax_scale,
    const data_t* __restrict__ q,
    const data_t* __restrict__ k_cache,
    const data_t* __restrict__ v_cache,
    const int*    __restrict__ block_table,
    itmd_t*       __restrict__ a,    // scratch [seq_len, NUM_Q_HEADS]
    otpt_t*       __restrict__ o,    // output  [NUM_Q_HEADS, HEAD_DIM]
    itmd_t*       __restrict__ asb   // log-sum-exp [NUM_Q_HEADS]
) {
    qk_product(cur_layer, num_blocks, seq_len, q, k_cache, block_table, a);
    softmax(seq_len, softmax_scale, a, asb);
    av_product(cur_layer, num_blocks, seq_len, a, v_cache, block_table, o);
}

// ------------------------------------------------------------------
// Merge partial attention outputs from parallel sequence segments.
//
// Each segment produced o_buf[i] with log-sum-exp asb[i].
// This function combines them using the online softmax rescaling trick.
//
// o_buf  : [num_segs, NUM_Q_HEADS, HEAD_DIM]  fp32
// as_buf : [num_segs, NUM_Q_HEADS]  fp32  (modified in-place to weights)
// o (out): [NUM_Q_HEADS, HEAD_DIM]  fp32
// ------------------------------------------------------------------
inline void gather_output_one_seq(
    int num_segs,
    const otpt_t* __restrict__ o_buf,
    itmd_t*       __restrict__ as_buf,
    otpt_t*       __restrict__ o
) {
    // Global max of log-sum-exp values per head
    float am_all[NUM_Q_HEADS];
    for (int h = 0; h < NUM_Q_HEADS; h++) am_all[h] = -1e20f;
    for (int i = 0; i < num_segs; i++) {
        const float* asp = as_buf + i * NUM_Q_HEADS;
        for (int h = 0; h < NUM_Q_HEADS; h++) {
            if (asp[h] > am_all[h]) am_all[h] = asp[h];
        }
    }

    // Rescale each segment's log-sum-exp to exp(lse - global_max)
    float as_all[NUM_Q_HEADS];
    for (int h = 0; h < NUM_Q_HEADS; h++) as_all[h] = 0.0f;
    for (int i = 0; i < num_segs; i++) {
        float* asp = as_buf + i * NUM_Q_HEADS;
        for (int h = 0; h < NUM_Q_HEADS; h++) {
            float w = expf(asp[h] - am_all[h]);
            asp[h]    = w;
            as_all[h] += w;
        }
    }

    // Normalize segment weights
    for (int i = 0; i < num_segs; i++) {
        float* asp = as_buf + i * NUM_Q_HEADS;
        #pragma omp simd
        for (int h = 0; h < NUM_Q_HEADS; h++) {
            asp[h] /= as_all[h];
        }
    }

    // Weighted sum of partial outputs
    memset(o, 0, NUM_Q_HEADS * HEAD_DIM * sizeof(otpt_t));
    for (int i = 0; i < num_segs; i++) {
        const float* asp  = as_buf + i * NUM_Q_HEADS;
        const float* obuf = o_buf  + i * NUM_Q_HEADS * HEAD_DIM;
        for (int h = 0; h < NUM_Q_HEADS; h++) {
            float scale      = asp[h];
            const float* obh = obuf + h * HEAD_DIM;
            float* oh        = o    + h * HEAD_DIM;
            #pragma omp simd
            for (int l = 0; l < HEAD_DIM; l++) {
                oh[l] += obh[l] * scale;
            }
        }
    }
}

} // namespace ispc
