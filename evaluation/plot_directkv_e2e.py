"""
DirectKV E2E Performance Figure Generator

Reads from:
  evaluation/estimated_fig10_directkv.csv   — Fig10 (online serving, TPOT vs rate)
  evaluation/estimated_fig11_directkv.csv   — Fig11 (long-ctx, kernel TPOT vs seq)
  results/fig11_longctx.csv                 — baseline comparisons (Neo, FlexGen, NoOffload)

Produces 5 figures in evaluation/figures/:
  fig_tpot_vs_rate.png       — Fig10-style: TPOT per token vs request rate
  fig_throughput_vs_rate.png — Fig10-style: throughput (tokens/s) vs request rate
  fig_kernel_tpot_vs_seq.png — Fig11-style: kernel decode latency vs seq length
  fig_hbm_kv_saved_gb.png    — KV bytes offloaded to CPU (GB) vs seq length
  fig_memory_saving_pct.png  — HBM savings % vs NoOffload baseline vs seq length

Usage:
  python evaluation/plot_directkv_e2e.py [--show]
"""

import argparse
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
COLORS = {
    "llama-3-8b": "#1f77b4",   # blue
    "opt-6.7b":   "#ff7f0e",   # orange
    "opt-30b":    "#2ca02c",   # green
    "Neo":        "#9467bd",   # purple
    "FlexGen":    "#8c564b",   # brown
    "NoOffload":  "#7f7f7f",   # grey
}
MARKERS = {
    "llama-3-8b": "o",
    "opt-6.7b":   "s",
    "opt-30b":    "^",
    "Neo":        "D",
    "FlexGen":    "v",
    "NoOffload":  "x",
}
DISPLAY_NAMES = {
    "llama-3-8b": "LLaMA-3.1-8B",
    "opt-6.7b":   "OPT-6.7B",
    "opt-30b":    "OPT-30B",
}

LINEWIDTH = 2.0
MARKERSIZE = 7
FIGSIZE_SINGLE = (6, 4)
FIGSIZE_WIDE   = (8, 4.5)
DPI = 150

# ---------------------------------------------------------------------------
# CSV readers
# ---------------------------------------------------------------------------

def read_fig10(path: Path):
    """Returns dict: model → {rate: {mean_latency, throughput_tps, ...}}"""
    data = {}
    if not path.exists():
        return data
    with open(path) as f:
        for row in csv.DictReader(f):
            model = row["model"]
            rate  = float(row["request_rate"])
            if model not in data:
                data[model] = {}
            data[model][rate] = {k: float(v) for k, v in row.items()
                                  if k not in ("system", "model", "dataset")}
    return data


def read_fig11(path: Path):
    """Returns dict: (system, model) → {seq_len: {mean_latency_s, ...}}"""
    data = {}
    if not path.exists():
        return data
    with open(path) as f:
        for row in csv.DictReader(f):
            system = row.get("system", "DirectKV")
            model  = row["model"]
            seq    = int(row["seq_length"])
            key    = (system, model)
            if key not in data:
                data[key] = {}
            data[key][seq] = {}
            for k, v in row.items():
                if k not in ("system", "model", "seq_length", "batch_size"):
                    try:
                        data[key][seq][k] = float(v)
                    except (ValueError, TypeError):
                        pass
    return data


# ---------------------------------------------------------------------------
# Figure 1: TPOT vs Request Rate
# ---------------------------------------------------------------------------

def plot_tpot_vs_rate(fig10: dict, out_path: Path, show: bool = False):
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    for model in ("llama-3-8b", "opt-6.7b", "opt-30b"):
        if model not in fig10:
            continue
        d = fig10[model]
        rates = sorted(d.keys())
        tpots_ms = [d[r]["mean_latency"] * 1000 for r in rates]   # s → ms
        label = DISPLAY_NAMES.get(model, model)
        ax.plot(rates, tpots_ms,
                color=COLORS[model], marker=MARKERS[model],
                linewidth=LINEWIDTH, markersize=MARKERSIZE,
                label=label)

    ax.set_xlabel("Request Rate (req/s)", fontsize=12)
    ax.set_ylabel("Mean TPOT (ms/token)", fontsize=12)
    ax.set_title("DirectKV: Inter-Token Latency vs Request Rate", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xticks([5, 10, 15, 20, 25, 30])
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI)
    print(f"  saved → {out_path}")
    if show:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Figure 2: Throughput vs Request Rate
# ---------------------------------------------------------------------------

def plot_throughput_vs_rate(fig10: dict, out_path: Path, show: bool = False):
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    for model in ("llama-3-8b", "opt-6.7b", "opt-30b"):
        if model not in fig10:
            continue
        d = fig10[model]
        rates = sorted(d.keys())
        tputs = [d[r]["throughput_tps"] for r in rates]
        label = DISPLAY_NAMES.get(model, model)
        ax.plot(rates, tputs,
                color=COLORS[model], marker=MARKERS[model],
                linewidth=LINEWIDTH, markersize=MARKERSIZE,
                label=label)

    ax.set_xlabel("Request Rate (req/s)", fontsize=12)
    ax.set_ylabel("Throughput (tokens/s)", fontsize=12)
    ax.set_title("DirectKV: Throughput vs Request Rate", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xticks([5, 10, 15, 20, 25, 30])

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI)
    print(f"  saved → {out_path}")
    if show:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Figure 3: Kernel TPOT vs Sequence Length (log scale)
# ---------------------------------------------------------------------------

def plot_kernel_tpot_vs_seq(fig11_directkv: dict, out_path: Path, show: bool = False):
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    for model in ("llama-3-8b", "opt-6.7b", "opt-30b"):
        key = ("DirectKV", model)
        if key not in fig11_directkv:
            continue
        d = fig11_directkv[key]
        seqs = sorted(d.keys())
        lats = [d[s]["mean_latency_s"] * 1000 for s in seqs]   # s → ms
        label = DISPLAY_NAMES.get(model, model)
        ax.plot(seqs, lats,
                color=COLORS[model], marker=MARKERS[model],
                linewidth=LINEWIDTH, markersize=MARKERSIZE,
                label=label)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Sequence Length (tokens)", fontsize=12)
    ax.set_ylabel("Decode Step Latency (ms)", fontsize=12)
    ax.set_title("DirectKV: Kernel TPOT vs Context Length (batch=1)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.set_xticks([1024, 2048, 4096, 8192])
    ax.set_xticklabels(["1K", "2K", "4K", "8K"])
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI)
    print(f"  saved → {out_path}")
    if show:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Figure 4: HBM KV Bytes Saved (GB)
# ---------------------------------------------------------------------------

def plot_hbm_saved_gb(fig11_directkv: dict, out_path: Path, show: bool = False):
    # KV bytes offloaded = kv_bytes_per_request (computed from model config)
    KV_BYTES = {
        "llama-3-8b": {1024: 0.134, 2048: 0.268, 4096: 0.537, 8192: 1.073},
        "opt-6.7b":   {1024: 0.537, 2048: 1.073, 4096: 2.147, 8192: 4.295},
        "opt-30b":    {1024: 1.409, 2048: 2.818, 4096: 5.637, 8192: 11.274},
    }

    seq_lens = [1024, 2048, 4096, 8192]
    x = np.arange(len(seq_lens))
    width = 0.25
    models = ("llama-3-8b", "opt-6.7b", "opt-30b")

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    for i, model in enumerate(models):
        vals = [KV_BYTES[model].get(s, 0) for s in seq_lens]
        bars = ax.bar(x + (i - 1) * width, vals, width,
                      label=DISPLAY_NAMES.get(model, model),
                      color=COLORS[model], alpha=0.85, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Sequence Length (tokens)", fontsize=12)
    ax.set_ylabel("KV Cache Offloaded to CPU (GB)", fontsize=12)
    ax.set_title("DirectKV: HBM Savings via KV Offloading", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(["1K", "2K", "4K", "8K"])
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI)
    print(f"  saved → {out_path}")
    if show:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Figure 5: Memory Saving % + NoOffload OOM comparison
# ---------------------------------------------------------------------------

def plot_memory_saving_pct(fig11_directkv: dict, fig11_longctx: dict,
                            out_path: Path, show: bool = False):
    seq_lens = [1024, 2048, 4096, 8192]
    x = np.arange(len(seq_lens))
    width = 0.35
    models = ("opt-6.7b", "opt-30b")

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    for i, model in enumerate(models):
        key = ("DirectKV", model)
        if key not in fig11_directkv:
            continue
        d = fig11_directkv[key]
        vals = []
        hatches = []
        for s in seq_lens:
            if s in d and d[s].get("memory_saving_pct", 0) == 100.0:
                vals.append(100.0)
                hatches.append("////")   # OOM case
            elif s in d:
                vals.append(d[s].get("memory_saving_pct", 0))
                hatches.append("")
            else:
                vals.append(0)
                hatches.append("")

        offset = (i - 0.5) * width
        for j, (val, hatch) in enumerate(zip(vals, hatches)):
            bar = ax.bar(x[j] + offset, val, width,
                         color=COLORS[model], alpha=0.85,
                         edgecolor="black", linewidth=0.5, hatch=hatch,
                         label=DISPLAY_NAMES.get(model, model) if j == 0 else "")
            if hatch:
                ax.text(x[j] + offset, val + 1.5, "OOM\navoided",
                        ha="center", va="bottom", fontsize=7, color="darkred")

    ax.set_xlabel("Sequence Length (tokens)", fontsize=12)
    ax.set_ylabel("HBM Freed vs NoOffload Baseline (%)", fontsize=12)
    ax.set_title("DirectKV: Memory Saving vs NoOffload (OPT Models)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(["1K", "2K", "4K", "8K"])
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS[m], label=DISPLAY_NAMES.get(m, m)) for m in models
    ]
    legend_elements.append(
        Patch(facecolor="white", edgecolor="black", hatch="////", label="OOM avoided")
    )
    ax.legend(handles=legend_elements, fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI)
    print(f"  saved → {out_path}")
    if show:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Bonus: Combined Fig11 comparison (DirectKV vs Neo vs FlexGen, LLaMA only)
# ---------------------------------------------------------------------------

def plot_fig11_comparison(fig11_longctx: dict, out_path: Path, show: bool = False):
    """Reproduce-style Fig11: decode latency comparison across systems for LLaMA."""
    model = "llama-3-8b"
    systems = ["DirectKV", "Neo", "Pie"]
    sys_colors = {"DirectKV": COLORS["llama-3-8b"], "Neo": COLORS["Neo"], "Pie": "#d62728"}
    sys_markers = {"DirectKV": "o", "Neo": "D", "Pie": "^"}

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    found_any = False
    for sys in systems:
        key = (sys, model)
        if key not in fig11_longctx:
            continue
        d = fig11_longctx[key]
        seqs = sorted(d.keys())
        lats_ms = [d[s]["mean_latency_s"] * 1000 for s in seqs]
        ax.plot(seqs, lats_ms,
                color=sys_colors.get(sys, "grey"),
                marker=sys_markers.get(sys, "o"),
                linewidth=LINEWIDTH, markersize=MARKERSIZE, label=sys)
        found_any = True

    if not found_any:
        plt.close()
        return

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sequence Length (tokens)", fontsize=12)
    ax.set_ylabel("Decode Step Latency (ms)", fontsize=12)
    ax.set_title("LLaMA-3.1-8B: System Comparison — Decode Latency vs Context", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.set_xticks([1024, 2048, 4096, 8192])
    ax.set_xticklabels(["1K", "2K", "4K", "8K"])

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI)
    print(f"  saved → {out_path}")
    if show:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--eval-dir", default=None,
                        help="evaluation/ directory (default: sibling of this script)")
    parser.add_argument("--results-dir", default=None,
                        help="results/ directory (default: AE-main/results/)")
    args = parser.parse_args()

    script_dir  = Path(__file__).parent
    eval_dir    = Path(args.eval_dir) if args.eval_dir else script_dir
    results_dir = Path(args.results_dir) if args.results_dir else script_dir.parent / "results"
    fig_dir     = eval_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    fig10     = read_fig10(eval_dir / "estimated_fig10_directkv.csv")
    fig11_est = read_fig11(eval_dir / "estimated_fig11_directkv.csv")
    fig11_ctx = read_fig11(results_dir / "fig11_longctx.csv")

    # Merge DirectKV entries from fig11_longctx into fig11_est if present
    for key, d in fig11_ctx.items():
        if key[0] == "DirectKV" and key not in fig11_est:
            fig11_est[key] = d

    print(f"\nGenerating figures in {fig_dir}/")
    plot_tpot_vs_rate(fig10,        fig_dir / "fig_tpot_vs_rate.png",       args.show)
    plot_throughput_vs_rate(fig10,  fig_dir / "fig_throughput_vs_rate.png", args.show)
    plot_kernel_tpot_vs_seq(fig11_est, fig_dir / "fig_kernel_tpot_vs_seq.png", args.show)
    plot_hbm_saved_gb(fig11_est,    fig_dir / "fig_hbm_kv_saved_gb.png",    args.show)
    plot_memory_saving_pct(fig11_est, fig11_ctx, fig_dir / "fig_memory_saving_pct.png", args.show)
    plot_fig11_comparison(fig11_ctx, fig_dir / "fig_system_comparison.png", args.show)

    print(f"\nDone. {len(list(fig_dir.glob('*.png')))} figures written to {fig_dir}/")


if __name__ == "__main__":
    main()
