#!/usr/bin/env python3
"""
Plot real microbenchmark results from bench_pie_vs_directkv.py and
bench_directkv.py runs on the actual H200 hardware.

Produces:
  - figure_micro_latency.pdf: Decode latency vs context length (3 backends)
  - figure_micro_crossover.pdf: Pie/DirectKV ratio showing exact crossover
  - figure_micro_memory.pdf: GPU memory comparison
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"GPU-only": "#7f7f7f", "DirectKV": "#1f77b4", "Pie": "#2ca02c"}

def plot_all(csv_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    seq_lens = df["seq_len"].values
    gpu_ms = df["gpu_only_ms"].values
    dkv_ms = df["directkv_ms"].values
    pie_ms = df["pie_ms"].values

    # --- Figure A: Latency vs Context Length ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(seq_lens, gpu_ms, "^-", color=COLORS["GPU-only"], linewidth=2,
            markersize=8, label="GPU-only (baseline)")
    ax.plot(seq_lens, dkv_ms, "o-", color=COLORS["DirectKV"], linewidth=2,
            markersize=8, label="DirectKV (CPU-pinned)")
    ax.plot(seq_lens, pie_ms, "s-", color=COLORS["Pie"], linewidth=2,
            markersize=8, label="Pie (async swap)")

    ax.set_xlabel("Context Length (tokens)", fontsize=12)
    ax.set_ylabel("Decode Latency (ms)", fontsize=12)
    import torch as _torch
    _gpu = _torch.cuda.get_device_name(0) if _torch.cuda.is_available() else "GPU"
    ax.set_title(f"Decode Attention Latency — {_gpu}\n"
                 "(32 layers, bs=4, nqh=32, nkvh=8, D=128, fp16)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xscale("log", base=2)

    path = os.path.join(output_dir, "figure_micro_latency.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path}")

    # --- Figure B: Crossover Ratio ---
    fig, ax = plt.subplots(figsize=(8, 5))

    pie_over_gpu = pie_ms / gpu_ms
    dkv_over_gpu = dkv_ms / gpu_ms
    pie_over_dkv = pie_ms / dkv_ms

    ax.plot(seq_lens, dkv_over_gpu, "o-", color=COLORS["DirectKV"], linewidth=2,
            markersize=8, label="DirectKV / GPU-only")
    ax.plot(seq_lens, pie_over_gpu, "s-", color=COLORS["Pie"], linewidth=2,
            markersize=8, label="Pie / GPU-only")

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Context Length (tokens)", fontsize=12)
    ax.set_ylabel("Latency Ratio vs GPU-only", fontsize=12)
    ax.set_title(f"DirectKV vs Pie — Crossover Analysis\n"
                 f"{_gpu}, 32 layers, bs=4",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xscale("log", base=2)

    # Annotate crossover
    for i in range(len(seq_lens) - 1):
        if dkv_over_gpu[i] < pie_over_gpu[i] and dkv_over_gpu[i+1] >= pie_over_gpu[i+1]:
            cross_x = (seq_lens[i] + seq_lens[i+1]) / 2
            ax.annotate(f"Crossover ~{int(cross_x)} tokens",
                        xy=(cross_x, 1.0), xytext=(cross_x, 2.0),
                        fontsize=10, color="purple",
                        ha="center", fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="purple"))
            break

    path = os.path.join(output_dir, "figure_micro_crossover.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path}")

    # --- Figure C: Stalls & Ratio Table ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    labels = [f"{s//1024}K" if s >= 1024 else str(s) for s in seq_lens]
    x = np.arange(len(labels))
    width = 0.3

    ax1.bar(x - width/2, pie_over_dkv, width, color=COLORS["Pie"],
            label="Pie / DirectKV ratio", edgecolor="white")
    ax1.axhline(y=1.0, color="gray", linestyle="--", linewidth=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("Context Length")
    ax1.set_ylabel("Latency Ratio (Pie / DirectKV)")
    ax1.set_title("(a) Pie vs DirectKV Ratio")
    ax1.grid(True, axis="y", alpha=0.3, linestyle="--")

    # Annotate which is better
    for i, ratio in enumerate(pie_over_dkv):
        color = COLORS["Pie"] if ratio < 1 else COLORS["DirectKV"]
        winner = "Pie" if ratio < 1 else "DKV"
        ax1.text(i, ratio + 0.05, f"{winner}\n{ratio:.2f}×",
                 ha="center", fontsize=8, color=color, fontweight="bold")

    # Stall count
    if "pie_stalls" in df.columns:
        stalls = df["pie_stalls"].values if "pie_stalls" in df.columns else df["ratio"].values * 0
    else:
        stalls = np.zeros(len(seq_lens))
    ax2.bar(x, stalls, width, color="#d62728", label="Pie stalls")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("Context Length")
    ax2.set_ylabel("Stall Count")
    ax2.set_title("(b) Pie Swap Stalls")
    ax2.grid(True, axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    path = os.path.join(output_dir, "figure_micro_comparison.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path}")


if __name__ == "__main__":
    csv_path = "results/microbench_pie_vs_directkv.csv"
    if os.path.exists(csv_path):
        plot_all(csv_path, "results/plots/")
    else:
        print(f"CSV not found: {csv_path}")
