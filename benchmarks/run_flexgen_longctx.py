#!/usr/bin/env python3
"""
FlexGen Long-Context Benchmark — Figure 11 data collection
===========================================================
Runs FlexGen's offline flex_opt inference for a range of sequence lengths
and appends rows to the unified Figure 11 CSV schema.

FlexGen has no HTTP server; this script drives it in-process by importing
the flex_opt module directly, capturing timing and GPU memory statistics.

Usage
-----
# OPT-6.7B, all-CPU KV placement:
python benchmarks/run_flexgen_longctx.py \
    --model facebook/opt-6.7b \
    --seq-lengths 512 1024 2048 \
    --batch-size 4 \
    --percent 0 100 100 0 100 0 \
    --output-csv results/fig11_longctx.csv

# Dry-run (print commands only):
python benchmarks/run_flexgen_longctx.py --model facebook/opt-1.3b --dry-run

Percent flags  (--percent W_gpu W_cpu KV_gpu KV_cpu Act_gpu Act_cpu)
    0 100 100 0 100 0  → weights on CPU, KV on GPU, acts on GPU
    0 100 0 100 100 0  → weights+KV on CPU (maximum GPU savings)
    100 0 100 0 100 0  → all on GPU (NoOffload baseline equivalent)

CSV schema (appended to existing file, header written if file is new):
  system, model, seq_length,
  mean_latency_s, p50_latency_s, p90_latency_s, p99_latency_s,
  gpu_memory_peak_mb, memory_saving_pct, throughput_tps
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

CSV_HEADER = [
    "system", "model", "seq_length",
    "mean_latency_s", "p50_latency_s", "p90_latency_s", "p99_latency_s",
    "gpu_memory_peak_mb", "memory_saving_pct", "throughput_tps",
]


def write_csv_row(path: str, row: Dict):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_HEADER})


# ---------------------------------------------------------------------------
# GPU memory polling
# ---------------------------------------------------------------------------

class GpuMemPoller:
    """Polls nvidia-smi every 0.2s in a background thread; records peak MiB."""

    def __init__(self, gpu_index: int = 0):
        self.gpu_index = gpu_index
        self._peak: float = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._stop.clear()
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        self._thread.join(timeout=5)
        return self._peak

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi",
                     f"--query-gpu=memory.used",
                     "--format=csv,noheader,nounits",
                     f"--id={self.gpu_index}"],
                    stderr=subprocess.DEVNULL,
                    timeout=3,
                ).decode().strip()
                val = float(out.split()[0])
                if val > self._peak:
                    self._peak = val
            except Exception:
                pass
            self._stop.wait(0.2)


# ---------------------------------------------------------------------------
# FlexGen runner
# ---------------------------------------------------------------------------

def run_flexgen_once(
    model: str,
    prompt_len: int,
    gen_len: int,
    gpu_batch_size: int,
    num_gpu_batches: int,
    percent: List[int],
    offload_dir: Optional[str],
    path: Optional[str] = None,
    gpu_index: int = 0,
) -> Tuple[float, float, float]:
    """
    Run one FlexGen configuration and return (latency_per_tok_s, peak_gpu_mb, throughput_tps).

    Drives flex_opt as a subprocess so it gets a fresh CUDA context each run
    and we can safely measure peak GPU memory.
    """
    cmd = [
        sys.executable, "-m", "flexllmgen.flex_opt",
        "--model", model,
        "--gpu-batch-size", str(gpu_batch_size),
        "--num-gpu-batches", str(num_gpu_batches),
        "--percent", *[str(p) for p in percent],
        "--prompt-len", str(prompt_len),
        "--gen-len", str(gen_len),
        "--verbose", "2",
    ]
    if path:
        cmd += ["--path", path]
    if offload_dir:
        cmd += ["--offload-dir", offload_dir]

    poller = GpuMemPoller(gpu_index=gpu_index)
    poller.start()
    t0 = time.perf_counter()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        wall_time = time.perf_counter() - t0
    except subprocess.TimeoutExpired:
        wall_time = time.perf_counter() - t0
        result = None
    finally:
        peak_mb = poller.stop()

    if result is None or result.returncode != 0:
        stderr = result.stderr[-2000:] if result else "timeout"
        print(f"  [FlexGen] FAILED (returncode={result.returncode if result else 'timeout'})")
        print(f"  stderr: {stderr}")
        return float("nan"), peak_mb, float("nan")

    # Parse total throughput from output.
    # flex_opt prints tab-separated pairs like:
    #   "decode latency: 0.154 s\tdecode throughput: 195.031 token/s"
    #   "total latency: 0.163 s\ttotal throughput: 196.123 token/s"
    # We split on tabs and look for the segment that contains "throughput", then
    # parse the first float in that segment to avoid picking up the latency value.
    throughput_tps = float("nan")
    for line in (result.stdout + result.stderr).splitlines():
        segments = line.split("\t")
        for seg in segments:
            seg_l = seg.lower()
            if "throughput" not in seg_l:
                continue
            if "token" not in seg_l:
                continue
            for part in seg.split():
                try:
                    val = float(part)
                    throughput_tps = val   # keep updating; last matching segment wins
                    break
                except ValueError:
                    pass

    total_tokens = gpu_batch_size * num_gpu_batches * gen_len
    if not np.isnan(throughput_tps) and throughput_tps > 0:
        latency_per_tok = 1.0 / throughput_tps
    elif wall_time > 0:
        latency_per_tok = wall_time / max(total_tokens, 1)
        throughput_tps = total_tokens / wall_time
    else:
        latency_per_tok = float("nan")

    return latency_per_tok, peak_mb, throughput_tps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="FlexGen long-context benchmark (Figure 11)")
    p.add_argument("--model", default="facebook/opt-6.7b", help="HuggingFace OPT model ID")
    p.add_argument("--model-label", default=None, help="Short label for CSV (default: last part of model ID)")
    p.add_argument("--system-label", default="FlexGen", help="System name in CSV")
    p.add_argument("--seq-lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096],
                   help="Input prompt lengths to sweep")
    p.add_argument("--gen-len", type=int, default=128, help="Output generation length")
    p.add_argument("--batch-size", type=int, default=4, help="GPU batch size (per FlexGen batch)")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=None,
                   help="Sweep multiple batch sizes (overrides --batch-size; used for Fig 10 rate simulation)")
    p.add_argument("--num-gpu-batches", type=int, default=1, help="Number of GPU batches")
    p.add_argument("--percent", type=int, nargs=6, default=[0, 100, 0, 100, 100, 0],
                   metavar=("W_GPU", "W_CPU", "KV_GPU", "KV_CPU", "ACT_GPU", "ACT_CPU"),
                   help="Placement percentages: W_gpu W_cpu KV_gpu KV_cpu Act_gpu Act_cpu")
    p.add_argument("--path", default=None, help="Path to pre-downloaded FlexGen numpy weights (e.g. /home/ubuntu/opt_weights)")
    p.add_argument("--offload-dir", default=None, help="Disk offload directory (for disk placement)")
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--runs", type=int, default=1, help="Repetitions per seq_length (takes mean)")
    p.add_argument("--nooffload-csv", default=None,
                   help="Figure 11 CSV with NoOffload rows (to compute memory_saving_pct)")
    p.add_argument("--output-csv", default="results/fig11_longctx.csv")
    p.add_argument("--dry-run", action="store_true", help="Print commands without running")
    return p.parse_args()


def load_nooffload_peak(csv_path: Optional[str]) -> Dict[int, float]:
    """Load NoOffload peak GPU memory per seq_length from an existing CSV."""
    result: Dict[int, float] = {}
    if not csv_path or not os.path.exists(csv_path):
        return result
    import csv as _csv
    with open(csv_path) as f:
        reader = _csv.DictReader(f)
        for row in reader:
            if row.get("system") == "NoOffload":
                try:
                    sl = int(row["seq_length"])
                    mb = float(row["gpu_memory_peak_mb"])
                    result[sl] = mb
                except (KeyError, ValueError):
                    pass
    return result


def main():
    args = parse_args()
    model_label = args.model_label or args.model.split("/")[-1].lower()

    # --batch-sizes sweeps multiple batch sizes at a fixed seq_length (Fig 10 mode)
    batch_sizes = args.batch_sizes if args.batch_sizes else [args.batch_size]

    nooffload_peaks = load_nooffload_peak(args.nooffload_csv or args.output_csv)

    print(f"\n{'='*70}")
    print(f"FlexGen Benchmark: model={model_label}, system={args.system_label}")
    print(f"  percent={args.percent}, batch_sizes={batch_sizes}×{args.num_gpu_batches}")
    print(f"  seq_lengths={args.seq_lengths}, gen_len={args.gen_len}")
    print(f"  output_csv={args.output_csv}")
    print(f"{'='*70}\n")

    for seq_len in args.seq_lengths:
      for batch_size in batch_sizes:
        latencies = []
        peaks = []
        tps_list = []

        for run_i in range(args.runs):
            print(f"[FlexGen] seq_len={seq_len}, bs={batch_size}, run {run_i+1}/{args.runs}  ", end="", flush=True)

            if args.dry_run:
                cmd_str = (
                    f"python -m flexllmgen.flex_opt --model {args.model} "
                    f"--gpu-batch-size {batch_size} "
                    f"--num-gpu-batches {args.num_gpu_batches} "
                    f"--percent {' '.join(map(str, args.percent))} "
                    f"--prompt-len {seq_len} --gen-len {args.gen_len}"
                )
                print(f"\n  [dry-run] {cmd_str}")
                continue

            lat, peak, tps = run_flexgen_once(
                model=args.model,
                prompt_len=seq_len,
                gen_len=args.gen_len,
                gpu_batch_size=batch_size,
                num_gpu_batches=args.num_gpu_batches,
                percent=args.percent,
                offload_dir=args.offload_dir,
                path=args.path,
                gpu_index=args.gpu_index,
            )
            print(f"lat={lat*1000:.1f}ms/tok  peak={peak:.0f}MB  tps={tps:.1f}")
            latencies.append(lat)
            peaks.append(peak)
            tps_list.append(tps)

        if args.dry_run:
            continue

        valid_lats = [x for x in latencies if not np.isnan(x)]
        valid_tps  = [x for x in tps_list  if not np.isnan(x)]
        valid_peaks = [x for x in peaks if x > 0]

        mean_lat = float(np.mean(valid_lats)) if valid_lats else float("nan")
        p50_lat  = float(np.percentile(valid_lats, 50)) if valid_lats else float("nan")
        p90_lat  = float(np.percentile(valid_lats, 90)) if valid_lats else float("nan")
        p99_lat  = float(np.percentile(valid_lats, 99)) if valid_lats else float("nan")
        mean_tps = float(np.mean(valid_tps))  if valid_tps  else float("nan")
        peak_mb  = float(np.max(valid_peaks)) if valid_peaks else float("nan")

        # Memory saving vs NoOffload baseline
        nooffload_peak = nooffload_peaks.get(seq_len)
        if nooffload_peak and nooffload_peak > 0 and not np.isnan(peak_mb):
            mem_saving_pct = (1.0 - peak_mb / nooffload_peak) * 100.0
        else:
            mem_saving_pct = float("nan")

        row = {
            "system": args.system_label,
            "model": model_label,
            "seq_length": seq_len,
            "mean_latency_s": round(mean_lat, 5),
            "p50_latency_s":  round(p50_lat,  5),
            "p90_latency_s":  round(p90_lat,  5),
            "p99_latency_s":  round(p99_lat,  5),
            "gpu_memory_peak_mb": round(peak_mb, 1),
            "memory_saving_pct":  round(mem_saving_pct, 2) if not np.isnan(mem_saving_pct) else "",
            "throughput_tps": round(mean_tps, 2),
        }
        write_csv_row(args.output_csv, row)
        print(f"  [saved] → {args.output_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()
