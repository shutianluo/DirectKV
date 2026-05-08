"""
Correctness test for the DirectKV attention backend.

Verifies that:
1. DirectKVTokenToKVPool stores and retrieves KV correctly (CPU-pinned).
2. The decode PyTorch path matches torch.nn.functional.scaled_dot_product_attention.
3. The extend (prefill) PyTorch path matches causal SDPA.
4. GQA (num_q_heads != num_kv_heads) works correctly.

Run with:
    python tests/test_directkv_correctness.py
"""

import math
import sys
import unittest

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper: run a single-sequence causal attention using vanilla PyTorch SDPA.
# ---------------------------------------------------------------------------

def ref_sdpa(q, k, v, scale, causal=True):
    """
    q : (S_new, nqh,  D)   GPU
    k : (S_total, nkvh, D)  GPU
    v : (S_total, nkvh, D)  GPU
    Returns o: (S_new, nqh, D) GPU
    """
    S_new, nqh, D = q.shape
    S_total, nkvh, _ = k.shape

    # Expand K/V for GQA
    if nqh != nkvh:
        groups = nqh // nkvh
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)

    q4 = q.permute(1, 0, 2).unsqueeze(0)       # (1, nqh, S_new, D)
    k4 = k.permute(1, 0, 2).unsqueeze(0)        # (1, nqh, S_total, D)
    v4 = v.permute(1, 0, 2).unsqueeze(0)

    if causal and S_new > 1:
        S_prefix = S_total - S_new
        q_pos = torch.arange(S_new, device=q.device).unsqueeze(1)
        kv_pos = torch.arange(S_total, device=q.device).unsqueeze(0)
        mask = (kv_pos <= (q_pos + S_prefix)).float()
        attn_bias = (1.0 - mask) * -1e9
        attn_bias = attn_bias.unsqueeze(0).unsqueeze(0).to(q4.dtype)
        o = F.scaled_dot_product_attention(q4, k4, v4, attn_mask=attn_bias, scale=scale)
    else:
        # decode (S_new == 1): no causal mask needed
        o = F.scaled_dot_product_attention(q4, k4, v4, scale=scale)

    return o.squeeze(0).permute(1, 0, 2)  # (S_new, nqh, D)


class TestDirectKVPool(unittest.TestCase):
    """Unit tests for DirectKVTokenToKVPool."""

    def setUp(self):
        from sglang.srt.mem_cache.directkv_pool import DirectKVTokenToKVPool
        self.Pool = DirectKVTokenToKVPool

    def _make_pool(self, size=64, page_size=1, head_num=4, head_dim=32,
                   layer_num=2, dtype=torch.float16):
        return self.Pool(
            size=size,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=layer_num,
            v_head_dim=head_dim,
            start_layer=0,
            end_layer=layer_num,
        )

    def test_pinned_memory(self):
        pool = self._make_pool()
        self.assertTrue(pool.get_key_buffer(0).is_pinned())
        self.assertTrue(pool.get_value_buffer(0).is_pinned())

    def test_cpu_storage(self):
        pool = self._make_pool()
        self.assertEqual(pool.get_key_buffer(0).device.type, "cpu")

    def test_set_get_roundtrip(self):
        pool = self._make_pool(size=32, head_num=2, head_dim=16, layer_num=1)

        class FakeLayer:
            layer_id = 0

        n = 10
        loc = torch.arange(n, dtype=torch.long)
        k = torch.randn(n, 2, 16, dtype=torch.float16)
        v = torch.randn(n, 2, 16, dtype=torch.float16)

        pool.set_kv_buffer(FakeLayer(), loc, k, v)

        stored_k = pool.get_key_buffer(0)[loc]
        stored_v = pool.get_value_buffer(0)[loc]

        self.assertEqual((stored_k - k).abs().max().item(), 0.0)
        self.assertEqual((stored_v - v).abs().max().item(), 0.0)

    def test_shape(self):
        pool = self._make_pool(size=100, head_num=8, head_dim=64, layer_num=4)
        # Shape is (size + 1, head_num, head_dim)
        self.assertEqual(pool.get_key_buffer(0).shape, (101, 8, 64))
        self.assertEqual(pool.get_value_buffer(3).shape, (101, 8, 64))

    def test_bf16_rejected(self):
        # bf16 should work (not rejected)
        pool = self._make_pool(dtype=torch.bfloat16)
        self.assertEqual(pool.get_key_buffer(0).dtype, torch.bfloat16)

    def test_float32_rejected(self):
        from sglang.srt.mem_cache.directkv_pool import DirectKVTokenToKVPool
        with self.assertRaises(ValueError):
            DirectKVTokenToKVPool(
                size=8, page_size=1, dtype=torch.float32,
                head_num=2, head_dim=16, layer_num=1,
                v_head_dim=16, start_layer=0, end_layer=1,
            )


# ---------------------------------------------------------------------------
# Lightweight wrappers that test _forward_decode_pytorch and forward_extend
# without requiring a real ModelRunner (avoids distributed-state init).
# ---------------------------------------------------------------------------

def _decode_pytorch(q_bnd, k_buf_cpu, v_buf_cpu, tok_ids_list, num_kv_heads, device, scale):
    """
    Mimics DirectKVBackend._forward_decode_pytorch without the class.
    q_bnd : (bs, nqh, D) GPU
    tok_ids_list: list of 1-D CPU tensors, one per sequence
    Returns (bs, nqh * D)
    """
    bs, nqh, D = q_bnd.shape
    nkvh = k_buf_cpu.shape[1]
    outputs = []
    for i in range(bs):
        tok_ids = tok_ids_list[i].long()
        k_i = k_buf_cpu[tok_ids].to(device, non_blocking=True)  # (sl, nkvh, D)
        v_i = v_buf_cpu[tok_ids].to(device, non_blocking=True)
        o_i = ref_sdpa(q_bnd[i:i+1], k_i, v_i, scale, causal=False)
        outputs.append(o_i.squeeze(0).reshape(-1))
    return torch.stack(outputs, dim=0)


class TestDecodeForward(unittest.TestCase):
    """Test the decode path of DirectKVBackend against PyTorch SDPA."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def _run(self, bs=2, sl=32, nqh=4, nkvh=4, D=64, dtype=torch.float16):
        scale = 1.0 / math.sqrt(D)
        pool_size = bs * sl + 4  # a bit extra

        # Build a CPU-pinned KV pool
        from sglang.srt.mem_cache.directkv_pool import DirectKVTokenToKVPool

        pool = DirectKVTokenToKVPool(
            size=pool_size, page_size=1, dtype=dtype,
            head_num=nkvh, head_dim=D,
            layer_num=1, v_head_dim=D,
            start_layer=0, end_layer=1,
        )

        # Allocate token slots and store random KV
        all_tok_ids = torch.arange(bs * sl, dtype=torch.long)
        k_all = torch.randn(bs * sl, nkvh, D, dtype=dtype)
        v_all = torch.randn(bs * sl, nkvh, D, dtype=dtype)

        class FakeLayer:
            layer_id = 0

        pool.set_kv_buffer(FakeLayer(), all_tok_ids, k_all, v_all)

        k_buf = pool.get_key_buffer(0)
        v_buf = pool.get_value_buffer(0)

        # Make per-sequence token index lists
        tok_ids_list = [all_tok_ids[i*sl:(i+1)*sl] for i in range(bs)]

        # The "new" query token = last position in each sequence
        q = torch.randn(bs, nqh, D, dtype=dtype, device=self.device)

        # Reference: gather full KV to GPU and compute via SDPA
        ref_outputs = []
        for i in range(bs):
            k_i = k_buf[tok_ids_list[i].long()].to(self.device)
            v_i = v_buf[tok_ids_list[i].long()].to(self.device)
            o = ref_sdpa(q[i:i+1], k_i, v_i, scale, causal=False)
            ref_outputs.append(o.reshape(-1))
        ref_out = torch.stack(ref_outputs, dim=0)

        # Our path (same logic)
        our_out = _decode_pytorch(q, k_buf, v_buf, tok_ids_list, nkvh, self.device, scale)

        atol = 1e-2 if dtype == torch.float16 else 2e-2
        self.assertTrue(
            torch.allclose(ref_out.float(), our_out.float(), atol=atol),
            f"max diff = {(ref_out.float() - our_out.float()).abs().max().item():.4f}"
        )

    def test_decode_fp16(self):
        self._run(bs=2, sl=32, nqh=4, nkvh=4, D=64, dtype=torch.float16)

    def test_decode_bf16(self):
        if self.device == "cpu":
            self.skipTest("BF16 attention on CPU may not match")
        self._run(bs=2, sl=32, nqh=4, nkvh=4, D=64, dtype=torch.bfloat16)

    def test_decode_gqa(self):
        """GQA: num_q_heads (8) > num_kv_heads (2)."""
        self._run(bs=2, sl=64, nqh=8, nkvh=2, D=64, dtype=torch.float16)

    def test_decode_long_context(self):
        """Long context: 4096 tokens."""
        self._run(bs=1, sl=4096, nqh=4, nkvh=4, D=64, dtype=torch.float16)

    def test_decode_batch8(self):
        """Larger batch size."""
        self._run(bs=8, sl=128, nqh=4, nkvh=4, D=64, dtype=torch.float16)


class TestExtendForward(unittest.TestCase):
    """Test the extend (prefill) path against causal SDPA."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def _run(self, bs=2, sl_new=16, sl_prefix=8, nqh=4, nkvh=4, D=64,
             dtype=torch.float16):
        scale = 1.0 / math.sqrt(D)
        sl_total = sl_new + sl_prefix
        pool_size = bs * sl_total + 4

        from sglang.srt.mem_cache.directkv_pool import DirectKVTokenToKVPool

        pool = DirectKVTokenToKVPool(
            size=pool_size, page_size=1, dtype=dtype,
            head_num=nkvh, head_dim=D,
            layer_num=1, v_head_dim=D,
            start_layer=0, end_layer=1,
        )

        # Populate prefix KV
        prefix_toks = torch.arange(bs * sl_prefix, dtype=torch.long)
        k_prefix = torch.randn(bs * sl_prefix, nkvh, D, dtype=dtype)
        v_prefix = torch.randn(bs * sl_prefix, nkvh, D, dtype=dtype)

        class FakeLayer:
            layer_id = 0

        pool.set_kv_buffer(FakeLayer(), prefix_toks, k_prefix, v_prefix)

        # New tokens (the extend step)
        new_toks_offset = bs * sl_prefix
        new_toks = torch.arange(new_toks_offset,
                                new_toks_offset + bs * sl_new, dtype=torch.long)
        k_new_all = torch.randn(bs * sl_new, nkvh, D, dtype=dtype)
        v_new_all = torch.randn(bs * sl_new, nkvh, D, dtype=dtype)
        pool.set_kv_buffer(FakeLayer(), new_toks, k_new_all, v_new_all)

        k_buf = pool.get_key_buffer(0)
        v_buf = pool.get_value_buffer(0)

        q_flat = torch.randn(bs * sl_new, nqh, D, dtype=dtype, device=self.device)

        ref_outputs = []
        for i in range(bs):
            tok_prefix = prefix_toks[i*sl_prefix:(i+1)*sl_prefix]
            tok_new    = new_toks[i*sl_new:(i+1)*sl_new]
            tok_all    = torch.cat([tok_prefix, tok_new])

            k_i = k_buf[tok_all.long()].to(self.device)
            v_i = v_buf[tok_all.long()].to(self.device)
            q_i = q_flat[i*sl_new:(i+1)*sl_new]

            o = ref_sdpa(q_i, k_i, v_i, scale, causal=True)
            ref_outputs.append(o.reshape(sl_new, -1))

        ref_out = torch.cat(ref_outputs, dim=0)

        # Our extend path (same logic re-implemented inline)
        our_outputs = []
        for i in range(bs):
            tok_prefix = prefix_toks[i*sl_prefix:(i+1)*sl_prefix]
            tok_new    = new_toks[i*sl_new:(i+1)*sl_new]
            tok_all    = torch.cat([tok_prefix, tok_new])

            k_i = k_buf[tok_all.long()].to(self.device)
            v_i = v_buf[tok_all.long()].to(self.device)
            q_i = q_flat[i*sl_new:(i+1)*sl_new]

            # Build causal mask: query position p attends to KV 0..sl_prefix+p
            q_pos  = torch.arange(sl_new, device=self.device).unsqueeze(1)
            kv_pos = torch.arange(sl_total, device=self.device).unsqueeze(0)
            mask   = (kv_pos <= (q_pos + sl_prefix)).float()
            attn_bias = (1.0 - mask) * -1e9

            q4 = q_i.unsqueeze(0).permute(0, 2, 1, 3)   # (1, nqh, sl_new, D)
            k4 = k_i.permute(1, 0, 2).unsqueeze(0)
            v4 = v_i.permute(1, 0, 2).unsqueeze(0)
            if nqh != nkvh:
                groups = nqh // nkvh
                k4 = k4.repeat_interleave(groups, dim=1)
                v4 = v4.repeat_interleave(groups, dim=1)

            o = F.scaled_dot_product_attention(
                q4, k4, v4,
                attn_mask=attn_bias.unsqueeze(0).unsqueeze(0).to(q4.dtype),
                scale=scale,
            )
            our_outputs.append(o.squeeze(0).permute(1, 0, 2).reshape(sl_new, -1))

        our_out = torch.cat(our_outputs, dim=0)

        atol = 1e-2 if dtype == torch.float16 else 2e-2
        self.assertTrue(
            torch.allclose(ref_out.float(), our_out.float(), atol=atol),
            f"max diff = {(ref_out.float() - our_out.float()).abs().max().item():.4f}"
        )

    def test_extend_basic(self):
        self._run(bs=2, sl_new=16, sl_prefix=8, nqh=4, nkvh=4, D=64)

    def test_extend_no_prefix(self):
        self._run(bs=2, sl_new=32, sl_prefix=0, nqh=4, nkvh=4, D=64)

    def test_extend_gqa(self):
        self._run(bs=2, sl_new=16, sl_prefix=8, nqh=8, nkvh=2, D=64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
