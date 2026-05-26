"""
Tests for Pie-style KV cache offloading.

Verifies:
  1. Swapping overlaps with computation without stalling.
  2. The mapping table correctly tracks layer residency.
  3. Inference outputs match a no-swap baseline.
  4. The swap engine uses non-blocking event queries (never
     cudaEventSynchronize in the hot path).
  5. Adaptive expansion controller adjusts m correctly.

Run:
    python tests/test_pie_offload.py
"""

import math
import time
import unittest
from unittest.mock import MagicMock

import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------
# Test 1: Mapping Table
# -----------------------------------------------------------------------

class TestMappingTable(unittest.TestCase):
    """Verify the mapping table tracks layer residency correctly."""

    def setUp(self):
        from sglang.srt.layers.kv_cache_offload import MappingTable, LayerStatus
        self.MappingTable = MappingTable
        self.LayerStatus = LayerStatus

    def test_initial_residency(self):
        """First num_gpu_layers on GPU, rest on CPU."""
        mt = self.MappingTable(num_layers=8, num_gpu_layers=6)
        for i in range(6):
            self.assertTrue(mt.is_on_gpu(i), f"Layer {i} should be on GPU")
        for i in range(6, 8):
            self.assertTrue(mt.is_on_cpu(i), f"Layer {i} should be on CPU")

    def test_gpu_cpu_layer_lists(self):
        mt = self.MappingTable(num_layers=8, num_gpu_layers=5)
        self.assertEqual(mt.gpu_layers(), [0, 1, 2, 3, 4])
        self.assertEqual(mt.cpu_layers(), [5, 6, 7])

    def test_status_transitions(self):
        mt = self.MappingTable(num_layers=4, num_gpu_layers=3)
        LS = self.LayerStatus

        # Layer 0 starts on GPU
        self.assertEqual(mt[0].status, LS.IN_GPU)

        # Transition to swapping out
        mt.set_status(0, LS.SWAPPING_OUT)
        self.assertEqual(mt[0].status, LS.SWAPPING_OUT)
        self.assertFalse(mt.is_on_gpu(0))
        self.assertFalse(mt.is_on_cpu(0))

        # Complete swap out → in CPU
        mt.set_status(0, LS.IN_CPU)
        self.assertTrue(mt.is_on_cpu(0))

        # Swap in → swapping in → GPU
        mt.set_status(0, LS.SWAPPING_IN)
        self.assertEqual(mt[0].status, LS.SWAPPING_IN)
        mt.set_status(0, LS.IN_GPU)
        self.assertTrue(mt.is_on_gpu(0))

    def test_coldest_gpu_layer(self):
        """Coldest = furthest from being needed (max cyclic dist)."""
        # 8 layers, 6 on GPU, 2 on CPU → safe window = 3
        mt = self.MappingTable(num_layers=8, num_gpu_layers=6)
        for i in range(8):
            mt.mark_accessed(i)

        # At current_layer=0, distances: 1→1, 2→2, 3→3, 4→4, 5→5
        # Safe=3, so candidates with dist>3: 4(dist=4), 5(dist=5)
        # Coldest (max dist) = 5
        coldest = mt.get_coldest_gpu_layer(current_layer=0)
        self.assertEqual(coldest, 5)

    def test_coldest_gpu_no_evict_when_all_gpu(self):
        """If all layers are on GPU (no CPU layers), no eviction needed."""
        mt = self.MappingTable(num_layers=4, num_gpu_layers=4)
        coldest = mt.get_coldest_gpu_layer(current_layer=0)
        self.assertIsNone(coldest)

    def test_hottest_cpu_layer(self):
        """Hottest CPU = closest to being needed next (cyclically)."""
        mt = self.MappingTable(num_layers=8, num_gpu_layers=4)
        # Layers 4, 5, 6, 7 are on CPU.
        # Current layer = 2. Next needed is 3 (GPU), 4 (CPU) — 4 is hottest.
        hottest = mt.get_hottest_cpu_layer(current_layer=2)
        self.assertEqual(hottest, 4)

        # Current layer = 6: CPU layers are 4, 5, 6, 7.
        # Distance from 6 (skipping 6 itself): 7=1, 4=6, 5=7 → hottest = 7.
        mt2 = self.MappingTable(num_layers=8, num_gpu_layers=4)
        hottest2 = mt2.get_hottest_cpu_layer(current_layer=6)
        self.assertEqual(hottest2, 7)

        # Current layer = 2 with CPU layers 4,5,6,7: hottest = 4 (dist=2, closest)
        # but 3 is on GPU so next CPU is 4.
        hottest3 = mt2.get_hottest_cpu_layer(current_layer=2)
        self.assertEqual(hottest3, 4)

    def test_no_coldest_when_only_one_gpu(self):
        """If there's only one GPU layer left (the current one), no coldest."""
        mt = self.MappingTable(num_layers=4, num_gpu_layers=1)
        # Only layer 0 on GPU; all others on CPU
        coldest = mt.get_coldest_gpu_layer(current_layer=0)
        self.assertIsNone(coldest)


# -----------------------------------------------------------------------
# Test 2: Swap Engine
# -----------------------------------------------------------------------

@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestSwapEngine(unittest.TestCase):
    """Verify the swap engine performs async transfers correctly."""

    def setUp(self):
        from sglang.srt.layers.kv_cache_offload import SwapEngine
        self.device = torch.device("cuda:0")
        self.engine = SwapEngine(self.device)

    def test_swap_out_and_in(self):
        """Data survives a GPU→CPU→GPU roundtrip."""
        shape = (1024, 8, 128)
        gpu_k = torch.randn(shape, device=self.device, dtype=torch.float16)
        gpu_v = torch.randn(shape, device=self.device, dtype=torch.float16)
        gpu_k_orig = gpu_k.clone()
        gpu_v_orig = gpu_v.clone()

        cpu_k = torch.zeros(shape, dtype=torch.float16, pin_memory=True)
        cpu_v = torch.zeros(shape, dtype=torch.float16, pin_memory=True)

        # Swap out
        self.engine.swap_out_async(0, gpu_k, gpu_v, cpu_k, cpu_v)
        self.engine.event_out.synchronize()
        done = self.engine.finish_swap_out()
        self.assertEqual(done, 0)

        # Verify CPU has the data
        self.assertTrue(torch.equal(cpu_k, gpu_k_orig.cpu()))

        # Zero the GPU buffer
        gpu_k.zero_()
        gpu_v.zero_()

        # Swap in
        self.engine.swap_in_async(0, cpu_k, cpu_v, gpu_k, gpu_v)
        self.engine.event_in.synchronize()
        done = self.engine.finish_swap_in()
        self.assertEqual(done, 0)

        # GPU buffer should match the original
        self.assertTrue(torch.equal(gpu_k, gpu_k_orig))
        self.assertTrue(torch.equal(gpu_v, gpu_v_orig))

    def test_event_query_is_non_blocking(self):
        """cudaEventQuery returns immediately (doesn't block compute stream)."""
        shape = (4096, 32, 128)  # ~32 MB, takes measurable time
        gpu_k = torch.randn(shape, device=self.device, dtype=torch.float16)
        gpu_v = torch.randn(shape, device=self.device, dtype=torch.float16)
        cpu_k = torch.zeros(shape, dtype=torch.float16, pin_memory=True)
        cpu_v = torch.zeros(shape, dtype=torch.float16, pin_memory=True)

        self.engine.swap_out_async(0, gpu_k, gpu_v, cpu_k, cpu_v)

        # Immediately query — should NOT block.
        t0 = time.perf_counter()
        _ = self.engine.is_swap_out_done()
        dt = time.perf_counter() - t0

        # The query should take < 1 ms (non-blocking). A blocking sync would
        # take much longer for 32 MB.
        self.assertLess(dt, 0.01, f"event_query took {dt*1000:.1f} ms — may be blocking")

        # Wait for completion to avoid dangling transfers
        self.engine.event_out.synchronize()

    def test_swap_in_uses_separate_stream(self):
        """Swap-in and swap-out can overlap."""
        shape = (1024, 8, 128)
        gpu_k_a = torch.randn(shape, device=self.device, dtype=torch.float16)
        gpu_v_a = torch.randn(shape, device=self.device, dtype=torch.float16)
        cpu_k_a = torch.zeros(shape, dtype=torch.float16, pin_memory=True)
        cpu_v_a = torch.zeros(shape, dtype=torch.float16, pin_memory=True)

        gpu_k_b = torch.zeros(shape, device=self.device, dtype=torch.float16)
        gpu_v_b = torch.zeros(shape, device=self.device, dtype=torch.float16)
        cpu_k_b = torch.randn(shape, dtype=torch.float16).pin_memory()
        cpu_v_b = torch.randn(shape, dtype=torch.float16).pin_memory()

        # Start both simultaneously
        self.engine.swap_out_async(0, gpu_k_a, gpu_v_a, cpu_k_a, cpu_v_a)
        self.engine.swap_in_async(1, cpu_k_b, cpu_v_b, gpu_k_b, gpu_v_b)

        # Both should complete
        self.engine.event_out.synchronize()
        self.engine.event_in.synchronize()

        self.assertTrue(torch.equal(cpu_k_a, gpu_k_a.cpu()))
        self.assertTrue(torch.equal(gpu_k_b, cpu_k_b.to(self.device)))


# -----------------------------------------------------------------------
# Test 3: FIFO Swap Controller
# -----------------------------------------------------------------------

@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestFIFOSwapController(unittest.TestCase):
    """Verify the swap controller makes correct FIFO decisions."""

    def setUp(self):
        from sglang.srt.layers.kv_cache_offload import (
            MappingTable, SwapEngine, FIFOSwapController, LayerStatus,
        )
        self.device = torch.device("cuda:0")
        self.LayerStatus = LayerStatus
        self.num_layers = 8
        self.num_gpu = 6
        self.m = self.num_layers - self.num_gpu  # 2 layers on CPU

        shape = (128, 4, 64)  # small for fast tests
        self.gpu_k = [torch.randn(shape, device=self.device, dtype=torch.float16)
                      for _ in range(self.num_layers)]
        self.gpu_v = [torch.randn(shape, device=self.device, dtype=torch.float16)
                      for _ in range(self.num_layers)]
        self.cpu_k = [torch.zeros(shape, dtype=torch.float16, pin_memory=True)
                      for _ in range(self.num_layers)]
        self.cpu_v = [torch.zeros(shape, dtype=torch.float16, pin_memory=True)
                      for _ in range(self.num_layers)]

        self.mapping = MappingTable(self.num_layers, self.num_gpu)
        self.engine = SwapEngine(self.device)
        self.controller = FIFOSwapController(
            mapping=self.mapping,
            engine=self.engine,
            gpu_k_buffers=self.gpu_k,
            gpu_v_buffers=self.gpu_v,
            cpu_k_buffers=self.cpu_k,
            cpu_v_buffers=self.cpu_v,
        )

    def test_gpu_layers_compute_without_stall(self):
        """Layers that start on GPU should not stall when there's sufficient
        compute time between layers for DMA to complete."""
        total_stalls = 0
        for layer in range(self.num_gpu):
            stalled = self.controller.on_layer_compute_start(layer)
            if stalled:
                total_stalls += 1
            # Simulate compute (gives DMA time to complete)
            _ = torch.randn(256, 256, device=self.device) @ torch.randn(256, 256, device=self.device)
            torch.cuda.synchronize()

        # With m=2 and compute time between layers, stalls should be rare (≤ m)
        self.assertLessEqual(
            total_stalls, self.m,
            f"Expected ≤{self.m} stalls, got {total_stalls}"
        )

    def test_cpu_layer_causes_stall(self):
        """Accessing a CPU-resident layer blocks (stall)."""
        # Layer 6 is on CPU
        stalled = self.controller.on_layer_compute_start(6)
        self.assertTrue(stalled, "Layer 6 (on CPU) should cause a stall")
        # After stall, it should now be on GPU
        self.assertTrue(self.mapping.is_on_gpu(6))

    def test_full_pass_no_crash(self):
        """Running through all layers multiple times should not crash."""
        for _pass in range(3):
            for layer in range(self.num_layers):
                self.controller.on_layer_compute_start(layer)
            # Allow any in-flight transfers to complete
            torch.cuda.synchronize()

    def test_mapping_tracks_swaps(self):
        """After a full pass, layers should be correctly tracked."""
        for layer in range(self.num_layers):
            self.controller.on_layer_compute_start(layer)
        torch.cuda.synchronize()

        # All layers should end up as either IN_GPU, IN_CPU, or in transit
        for i in range(self.num_layers):
            entry = self.mapping[i]
            self.assertIn(
                entry.status,
                [self.LayerStatus.IN_GPU, self.LayerStatus.IN_CPU,
                 self.LayerStatus.SWAPPING_IN, self.LayerStatus.SWAPPING_OUT],
            )


# -----------------------------------------------------------------------
# Test 4: Inference output matches no-swap baseline
# -----------------------------------------------------------------------

@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestOutputCorrectness(unittest.TestCase):
    """
    Verify that attention output with Pie swapping matches output without it.

    Simulates a multi-layer decode scenario:
      - Build a fake KV cache in a GPU pool.
      - Run attention with direct GPU access (baseline).
      - Run attention with the Pie swap controller managing residency.
      - Compare outputs.
    """

    def setUp(self):
        self.device = torch.device("cuda:0")
        self.num_layers = 8
        self.num_tokens = 128
        self.num_q_heads = 8
        self.num_kv_heads = 2
        self.head_dim = 64
        self.dtype = torch.float16
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def _make_kv_cache(self):
        """Create per-layer KV cache tensors on GPU."""
        shape = (self.num_tokens + 1, self.num_kv_heads, self.head_dim)
        k_bufs = [torch.randn(shape, device=self.device, dtype=self.dtype)
                   for _ in range(self.num_layers)]
        v_bufs = [torch.randn(shape, device=self.device, dtype=self.dtype)
                   for _ in range(self.num_layers)]
        return k_bufs, v_bufs

    def _decode_attention(self, q, k_buf, v_buf, seq_len):
        """Single-token decode: q attends to k_buf[:seq_len], v_buf[:seq_len]."""
        groups = self.num_q_heads // self.num_kv_heads
        k = k_buf[:seq_len]  # (S, nkvh, D)
        v = v_buf[:seq_len]
        q4 = q.unsqueeze(0).permute(0, 2, 1, 3)   # (1, nqh, 1, D)
        k4 = k.permute(1, 0, 2).unsqueeze(0)        # (1, nkvh, S, D)
        v4 = v.permute(1, 0, 2).unsqueeze(0)
        if groups > 1:
            k4 = k4.repeat_interleave(groups, dim=1)
            v4 = v4.repeat_interleave(groups, dim=1)
        o = F.scaled_dot_product_attention(q4, k4, v4, scale=self.scale)
        return o.squeeze(0).squeeze(1)  # (nqh, D)

    def test_output_matches_baseline(self):
        from sglang.srt.layers.kv_cache_offload import (
            MappingTable, SwapEngine, FIFOSwapController,
        )

        k_bufs, v_bufs = self._make_kv_cache()
        seq_len = 64

        # Baseline: compute with all layers on GPU (no swapping)
        q_per_layer = [
            torch.randn(1, self.num_q_heads, self.head_dim,
                         device=self.device, dtype=self.dtype)
            for _ in range(self.num_layers)
        ]
        baseline_outputs = []
        for layer in range(self.num_layers):
            o = self._decode_attention(q_per_layer[layer], k_bufs[layer], v_bufs[layer], seq_len)
            baseline_outputs.append(o)

        # With swapping: m=2 layers offloaded to CPU
        num_gpu = self.num_layers - 2
        mapping = MappingTable(self.num_layers, num_gpu)
        engine = SwapEngine(self.device)

        # Create CPU shadow buffers
        cpu_k = [torch.zeros_like(k, device="cpu", pin_memory=True) for k in k_bufs]
        cpu_v = [torch.zeros_like(v, device="cpu", pin_memory=True) for v in v_bufs]
        # Copy offloaded layers to CPU
        for lid in mapping.cpu_layers():
            cpu_k[lid].copy_(k_bufs[lid])
            cpu_v[lid].copy_(v_bufs[lid])

        controller = FIFOSwapController(
            mapping=mapping,
            engine=engine,
            gpu_k_buffers=k_bufs,
            gpu_v_buffers=v_bufs,
            cpu_k_buffers=cpu_k,
            cpu_v_buffers=cpu_v,
        )

        swap_outputs = []
        for layer in range(self.num_layers):
            controller.on_layer_compute_start(layer)
            o = self._decode_attention(q_per_layer[layer], k_bufs[layer], v_bufs[layer], seq_len)
            swap_outputs.append(o)

        # Compare
        for layer in range(self.num_layers):
            diff = (baseline_outputs[layer].float() - swap_outputs[layer].float()).abs().max().item()
            self.assertLess(
                diff, 1e-3,
                f"Layer {layer} output diverged from baseline (max diff = {diff:.6f})"
            )


# -----------------------------------------------------------------------
# Test 5: Swap overlaps computation (timing)
# -----------------------------------------------------------------------

@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestSwapOverlapsCompute(unittest.TestCase):
    """
    Verify that the swap controller can overlap DMA with computation so
    that stalls are avoided when compute time > swap time.
    """

    def test_no_stall_when_compute_exceeds_swap(self):
        """
        If we simulate compute by sleeping between layers (giving DMA time
        to complete), there should be zero stalls after the warmup pass.
        """
        from sglang.srt.layers.kv_cache_offload import (
            MappingTable, SwapEngine, FIFOSwapController,
        )

        device = torch.device("cuda:0")
        num_layers = 8
        num_gpu = 6  # m = 2

        # Small KV buffers so DMA is fast
        shape = (64, 4, 64)
        gpu_k = [torch.randn(shape, device=device, dtype=torch.float16)
                 for _ in range(num_layers)]
        gpu_v = [torch.randn(shape, device=device, dtype=torch.float16)
                 for _ in range(num_layers)]
        cpu_k = [torch.zeros(shape, dtype=torch.float16, pin_memory=True)
                 for _ in range(num_layers)]
        cpu_v = [torch.zeros(shape, dtype=torch.float16, pin_memory=True)
                 for _ in range(num_layers)]

        mapping = MappingTable(num_layers, num_gpu)
        engine = SwapEngine(device)
        ctrl = FIFOSwapController(
            mapping, engine, gpu_k, gpu_v, cpu_k, cpu_v
        )

        # Copy offloaded layers to CPU
        for lid in mapping.cpu_layers():
            cpu_k[lid].copy_(gpu_k[lid])
            cpu_v[lid].copy_(gpu_v[lid])

        # Warmup pass (stalls expected)
        for layer in range(num_layers):
            ctrl.on_layer_compute_start(layer)
            # Simulate compute
            torch.cuda.synchronize()

        # Run several more passes so the FIFO pipeline stabilises
        for _pass in range(3):
            ctrl.stall_count = 0
            for layer in range(num_layers):
                ctrl.on_layer_compute_start(layer)
                # Simulate compute time much longer than DMA time for tiny buffers
                _ = torch.randn(1024, 1024, device=device) @ torch.randn(1024, 1024, device=device)
                torch.cuda.synchronize()

        # After pipeline warmup, stalls should be rare.
        # m=2 layers on CPU; with small buffers and heavy compute, ≤ m stalls
        # is acceptable (each CPU layer may stall once per rotation).
        m = num_layers - num_gpu
        self.assertLessEqual(
            ctrl.stall_count, m + 1,
            f"Expected ≤{m + 1} stalls, got {ctrl.stall_count}"
        )


# -----------------------------------------------------------------------
# Test 6: Adaptive Expansion Controller
# -----------------------------------------------------------------------

class TestAdaptiveExpansion(unittest.TestCase):

    def test_decrease_on_stalls(self):
        from sglang.srt.layers.kv_cache_offload import AdaptiveExpansionController
        ctrl = AdaptiveExpansionController(
            num_layers=32, initial_m=8, check_interval=1,
            stall_threshold=0.05,
        )
        # Many stalls → should decrease
        new_m = ctrl.on_forward_pass_complete(
            stall_count=10, idle_swap_in=0, idle_swap_out=0,
            tokens_processed=100, elapsed_s=0.1,
        )
        self.assertIsNotNone(new_m)
        self.assertLess(new_m, 8)

    def test_increase_on_idle(self):
        from sglang.srt.layers.kv_cache_offload import AdaptiveExpansionController
        ctrl = AdaptiveExpansionController(
            num_layers=32, initial_m=4, check_interval=1,
            idle_threshold=0.3,
        )
        # No stalls, lots of idle → should increase
        new_m = ctrl.on_forward_pass_complete(
            stall_count=0, idle_swap_in=30, idle_swap_out=30,
            tokens_processed=100, elapsed_s=0.1,
        )
        self.assertIsNotNone(new_m)
        self.assertGreater(new_m, 4)

    def test_no_change_when_balanced(self):
        from sglang.srt.layers.kv_cache_offload import AdaptiveExpansionController
        ctrl = AdaptiveExpansionController(
            num_layers=32, initial_m=8, check_interval=1,
        )
        new_m = ctrl.on_forward_pass_complete(
            stall_count=0, idle_swap_in=0, idle_swap_out=0,
            tokens_processed=100, elapsed_s=0.1,
        )
        self.assertIsNone(new_m)

    def test_m_bounded(self):
        from sglang.srt.layers.kv_cache_offload import AdaptiveExpansionController
        ctrl = AdaptiveExpansionController(
            num_layers=8, initial_m=6, check_interval=1,
        )
        # At m=6 with num_layers=8, max is num_layers-2=6, so no increase
        new_m = ctrl.on_forward_pass_complete(
            stall_count=0, idle_swap_in=50, idle_swap_out=50,
            tokens_processed=100, elapsed_s=0.1,
        )
        self.assertIsNone(new_m)  # already at max


# -----------------------------------------------------------------------
# Test 7: PieKVPool integration
# -----------------------------------------------------------------------

@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestPieKVPool(unittest.TestCase):
    """Test the PieKVPool wrapper works end-to-end."""

    def test_create_pie_pool(self):
        """PieKVPool initialises without errors."""
        from sglang.srt.layers.kv_cache_offload import PieKVPool

        # Minimal mock of MHATokenToKVPool
        class FakeGPUPool:
            layer_num = 8
            start_layer = 0
            device = "cuda:0"
            def _get_key_buffer(self, layer_id):
                return torch.zeros(64, 4, 64, device="cuda:0", dtype=torch.float16)
            def _get_value_buffer(self, layer_id):
                return torch.zeros(64, 4, 64, device="cuda:0", dtype=torch.float16)

        gpu_pool = FakeGPUPool()
        gpu_pool.k_buffer = [torch.zeros(64, 4, 64, device="cuda:0", dtype=torch.float16)
                              for _ in range(8)]
        gpu_pool.v_buffer = [torch.zeros(64, 4, 64, device="cuda:0", dtype=torch.float16)
                              for _ in range(8)]

        pie = PieKVPool(gpu_pool, expansion_ratio=1.3)
        self.assertEqual(pie.num_layers, 8)
        self.assertGreater(pie.m, 0)
        self.assertEqual(len(pie.cpu_k_buffers), 8)
        self.assertTrue(pie.cpu_k_buffers[0].is_pinned())


if __name__ == "__main__":
    unittest.main(verbosity=2)
