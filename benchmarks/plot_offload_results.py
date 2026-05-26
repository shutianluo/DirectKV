#!/usr/bin/env python3
"""
Plotting Scripts for Offload Benchmark Results
================================================
Produces:
  - Figure 10: Latency vs Request Rate (3 subplots by model, line plots)
  - Figure 11: Long-Context Performance (bar charts: latency + memory saving)
  - Figure 12: DirectKV vs Pie Crossover Analysis

Usage:
    python benchmarks/plot_offload_results.py \\
        --fig10-csv results/fig10_latency_vs_rate.csv \\
        --fig11-csv results/fig11_longctx.csv \\
        --output-dir results/plots/

Requires: matplotlib, pandas, numpy
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Style constants (match comparison.md exactly)
# ---------------------------------------------------------------------------

COLORS = {
    "DirectKV": "#1f77b4",   # blue
    "Neo": "#ff7f0e",        # orange
    "Pie": "#2ca02c",        # green
    "FlexGen": "#d62728",    # red
    "NoOffload": "#7f7f7f",  # gray (baseline)
}

MARKERS = {
    "DirectKV": "o",   # circle
    "Neo": "x",        # x
    "Pie": "s",        # square
    "FlexGen": "D",    # diamond
    "NoOffload": "^",  # triangle
}

LINE_STYLES = {
    "DirectKV": "-",
    "Neo": "--",
    "Pie": "-.",
    "FlexGen": ":",
    "NoOffload": "-",
}

SYSTEMS_ORDER = ["DirectKV", "Pie", "Neo", "FlexGen"]
SYSTEMS_WITH_BASELINE = ["NoOffload"] + SYSTEMS_ORDER

MODEL_TITLES = {
    "llama-3.1-8b": "LLaMA-3.1-8B",
    "llama-3-8b": "LLaMA-3-8B",
    "opt-6.7b": "OPT-6.7B",
    "opt-30b": "OPT-30B",
}


# ---------------------------------------------------------------------------
# Figure 10 — Latency vs Request Rate (line plots, 3 subplots by model)
# ---------------------------------------------------------------------------

def plot_figure10(csv_path: str, output_path: str = "figure10.pdf"):
    """
    3 subplots (one per model). X = request rate, Y = per-token latency.
    4 lines per subplot: DirectKV, Pie, Neo, FlexGen.
    """
    df = pd.read_csv(csv_path)
    # Normalize system names: DirectKV_OPT6B → DirectKV, Pie_OPT6B → Pie, etc.
    df["system"] = df["system"].str.replace(r"_(OPT6B|OPT30B|LLaMA)$", "", regex=True)

    # Use e2e_latency / output_len as the y-axis: this captures queue wait
    # (TPOT alone is decode-only and stays flat under saturation, which
    # erases the "hockey-stick" shape we need to reproduce).
    if "mean_e2e_latency" in df.columns and "output_len" in df.columns:
        # Some rows may have output_len=0 if the bench couldn't read it; fall
        # back to model-specific defaults so plotting still works.
        _OUT_DEFAULTS = {"llama-3.1-8b": 256, "llama-3-8b": 256,
                          "opt-6.7b": 200, "opt-30b": 256}
        df["_out"] = df.apply(
            lambda r: r["output_len"] if r.get("output_len", 0) > 0
            else _OUT_DEFAULTS.get(r["model"], 256),
            axis=1,
        )
        df["_y"] = df["mean_e2e_latency"] / df["_out"]
    else:
        # CSV from an older bench without e2e columns — fall back to TPOT.
        df["_y"] = df["mean_latency"]

    models = sorted(df["model"].unique())

    # Use 3 subplots or fewer depending on data
    n_models = len(models)
    fig, axes = plt.subplots(1, max(n_models, 1), figsize=(5 * n_models, 4.5),
                             squeeze=False)
    axes = axes[0]

    for ax_i, model in enumerate(models):
        ax = axes[ax_i]
        model_df = df[df["model"] == model]

        for system in SYSTEMS_ORDER:
            subset = model_df[model_df["system"] == system].sort_values("request_rate")
            if subset.empty:
                continue
            ax.plot(
                subset["request_rate"],
                subset["_y"],
                marker=MARKERS.get(system, "o"),
                color=COLORS.get(system, "#333333"),
                linestyle=LINE_STYLES.get(system, "-"),
                linewidth=2,
                markersize=7,
                label=system,
            )

        ax.set_xlabel("Request Rate (req/s)", fontsize=11)
        title = MODEL_TITLES.get(model, model)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.tick_params(labelsize=10)

    axes[0].set_ylabel("Per-Token Latency (s)", fontsize=11)
    # Single legend for the figure
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(SYSTEMS_ORDER),
                   fontsize=10, frameon=True, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Figure 10 saved to {output_path}")


# ---------------------------------------------------------------------------
# Figure 11 — Long-Context Performance (grouped bar charts)
# ---------------------------------------------------------------------------

def plot_figure11(csv_path: str, output_path: str = "figure11.pdf"):
    """
    2 subplots side by side:
      (a) Per-token latency: X = seq length, Y = latency (s), 4 grouped bars
      (b) GPU Memory Saving: X = seq length, Y = memory saving (%), 4 grouped bars
    """
    df = pd.read_csv(csv_path)
    # Normalize system names: DirectKV_OPT6B → DirectKV, Pie_OPT6B → Pie, etc.
    df["system"] = df["system"].str.replace(r"_(OPT6B|OPT30B|LLaMA)$", "", regex=True)
    # Exclude NoOffload from the bar charts (it's the baseline)
    df_plot = df[df["system"].isin(SYSTEMS_ORDER)]

    seq_lengths = sorted(df_plot["seq_length"].unique())
    labels = [f"{sl//1024}K" if sl >= 1024 else str(sl) for sl in seq_lengths]
    x = np.arange(len(seq_lengths))
    n_systems = len(SYSTEMS_ORDER)
    width = 0.8 / n_systems

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for i, system in enumerate(SYSTEMS_ORDER):
        subset = df_plot[df_plot["system"] == system].sort_values("seq_length")
        if subset.empty:
            continue

        # Match seq_lengths ordering
        latencies = []
        mem_savings = []
        for sl in seq_lengths:
            row = subset[subset["seq_length"] == sl]
            if not row.empty:
                latencies.append(row.iloc[0]["mean_latency_s"])
                mem_savings.append(row.iloc[0].get("memory_saving_pct", 0))
            else:
                latencies.append(0)
                mem_savings.append(0)

        offset = (i - n_systems / 2 + 0.5) * width
        ax1.bar(x + offset, latencies, width,
                label=system, color=COLORS.get(system, "#333333"),
                edgecolor="white", linewidth=0.5)
        ax2.bar(x + offset, mem_savings, width,
                label=system, color=COLORS.get(system, "#333333"),
                edgecolor="white", linewidth=0.5)

    # Format axes
    for ax, ylabel, title in [
        (ax1, "Per-Token Latency (s)", "(a) Latency"),
        (ax2, "GPU Memory Saving (%)", "(b) Memory Saving"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_xlabel("Sequence Length", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.tick_params(labelsize=10)

    # Legend
    handles, labels_leg = ax1.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_leg, loc="upper center", ncol=n_systems,
                   fontsize=10, frameon=True, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Figure 11 saved to {output_path}")


# ---------------------------------------------------------------------------
# Figure 12 — DirectKV vs Pie Crossover Analysis
# ---------------------------------------------------------------------------

def plot_crossover(
    fig10_csv: str,
    fig11_csv: str,
    output_path: str = "figure12_crossover.pdf",
):
    """
    Identifies and visualises the crossover point where Pie overtakes DirectKV.

    (a) Latency ratio (Pie / DirectKV) vs request rate — per model.
        Ratio < 1 means Pie is faster.
    (b) Latency ratio vs sequence length.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # -- (a) vs request rate ---
    if os.path.exists(fig10_csv):
        df = pd.read_csv(fig10_csv)
        df["system"] = df["system"].str.replace(r"_(OPT6B|OPT30B|LLaMA)$", "", regex=True)
        models = sorted(df["model"].unique())
        model_colors = plt.cm.tab10(np.linspace(0, 0.4, max(len(models), 1)))

        for mi, model in enumerate(models):
            dkv = df[(df["system"] == "DirectKV") & (df["model"] == model)].sort_values("request_rate")
            pie = df[(df["system"] == "Pie") & (df["model"] == model)].sort_values("request_rate")
            if dkv.empty or pie.empty:
                continue

            merged = pd.merge(dkv, pie, on="request_rate", suffixes=("_dkv", "_pie"))
            ratio = merged["mean_latency_pie"] / merged["mean_latency_dkv"]

            ax1.plot(merged["request_rate"], ratio,
                     marker="o", linewidth=2, markersize=6,
                     color=model_colors[mi],
                     label=MODEL_TITLES.get(model, model))

        ax1.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax1.fill_between(ax1.get_xlim(), 0, 1, alpha=0.05, color="green",
                         label="_Pie wins")
        ax1.set_xlabel("Request Rate (req/s)", fontsize=11)
        ax1.set_ylabel("Latency Ratio (Pie / DirectKV)", fontsize=11)
        ax1.set_title("(a) Crossover vs Request Rate", fontsize=12, fontweight="bold")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3, linestyle="--")

        # Annotate crossover
        ax1.annotate("Pie faster\n(ratio < 1)", xy=(0.95, 0.05),
                     xycoords="axes fraction", ha="right", fontsize=9,
                     color="green", fontstyle="italic")
        ax1.annotate("DirectKV faster\n(ratio > 1)", xy=(0.95, 0.95),
                     xycoords="axes fraction", ha="right", va="top", fontsize=9,
                     color="blue", fontstyle="italic")

    # -- (b) vs sequence length ---
    if os.path.exists(fig11_csv):
        df = pd.read_csv(fig11_csv)
        df["system"] = df["system"].str.replace(r"_(OPT6B|OPT30B|LLaMA)$", "", regex=True)
        dkv = df[df["system"] == "DirectKV"].sort_values("seq_length")
        pie = df[df["system"] == "Pie"].sort_values("seq_length")

        if not dkv.empty and not pie.empty:
            merged = pd.merge(dkv, pie, on="seq_length", suffixes=("_dkv", "_pie"))
            ratio = merged["mean_latency_s_pie"] / merged["mean_latency_s_dkv"]
            labels = [f"{sl//1024}K" if sl >= 1024 else str(sl)
                      for sl in merged["seq_length"]]

            ax2.bar(range(len(labels)), ratio, color="#9467bd", edgecolor="white")
            ax2.axhline(y=1.0, color="gray", linestyle="--", linewidth=1)
            ax2.set_xticks(range(len(labels)))
            ax2.set_xticklabels(labels, fontsize=11)

    ax2.set_xlabel("Sequence Length", fontsize=11)
    ax2.set_ylabel("Latency Ratio (Pie / DirectKV)", fontsize=11)
    ax2.set_title("(b) Crossover vs Sequence Length", fontsize=12, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Figure 12 (crossover) saved to {output_path}")


# ---------------------------------------------------------------------------
# Generate sample data for testing the plots
# ---------------------------------------------------------------------------

def generate_sample_data(output_dir: str):
    """Create sample CSVs for testing the plotting scripts without a live server."""
    os.makedirs(output_dir, exist_ok=True)

    # -- Figure 10 sample data --
    fig10_path = os.path.join(output_dir, "fig10_latency_vs_rate.csv")
    rates = [5, 10, 15, 20, 25, 30]
    models = ["llama-3.1-8b", "opt-6.7b", "opt-30b"]

    # Latency profiles (system → base_latency, rate_sensitivity)
    profiles = {
        "DirectKV":  {"base": 0.045, "rate_sens": 0.0015, "model_scale": {"llama-3.1-8b": 1.0, "opt-6.7b": 1.1, "opt-30b": 1.8}},
        "Pie":       {"base": 0.040, "rate_sens": 0.0010, "model_scale": {"llama-3.1-8b": 1.0, "opt-6.7b": 1.1, "opt-30b": 1.7}},
        "Neo":       {"base": 0.065, "rate_sens": 0.0025, "model_scale": {"llama-3.1-8b": 1.0, "opt-6.7b": 1.15, "opt-30b": 2.0}},
        "FlexGen":   {"base": 0.120, "rate_sens": 0.0060, "model_scale": {"llama-3.1-8b": 1.0, "opt-6.7b": 1.2, "opt-30b": 2.5}},
    }

    rows = []
    rng = np.random.RandomState(42)
    for model in models:
        for system, prof in profiles.items():
            for rate in rates:
                scale = prof["model_scale"][model]
                base = prof["base"] * scale
                mean_lat = base + prof["rate_sens"] * rate * scale
                noise = rng.normal(0, 0.003)
                mean_lat = max(0.01, mean_lat + noise)
                p50 = mean_lat * 0.90
                p90 = mean_lat * 1.40
                p99 = mean_lat * 2.00
                tps = 1000 / mean_lat * (1 - rate * 0.01)  # rough approx

                rows.append({
                    "system": system, "model": model, "dataset": "sharegpt",
                    "request_rate": rate,
                    "mean_latency": round(mean_lat, 4),
                    "p50_latency": round(p50, 4),
                    "p90_latency": round(p90, 4),
                    "p99_latency": round(p99, 4),
                    "throughput_tps": round(tps, 0),
                    "mean_ttft": round(mean_lat * 0.3, 4),
                    "p99_ttft": round(mean_lat * 0.8, 4),
                    "num_completed": 900, "num_failed": 0,
                })

    df10 = pd.DataFrame(rows)
    df10.to_csv(fig10_path, index=False)
    print(f"[sample] Figure 10 data → {fig10_path}")

    # -- Figure 11 sample data --
    fig11_path = os.path.join(output_dir, "fig11_longctx.csv")
    seq_lengths = [1024, 2048, 4096, 8192]

    # Memory baseline (NoOffload)
    baseline_mem = {1024: 45000, 2048: 55000, 4096: 70000, 8192: 90000}

    # Profiles: (latency_per_1k_tokens, memory_fraction_of_baseline)
    longctx_profiles = {
        "NoOffload": {"lat_per_1k": 0.8, "mem_frac": 1.00},
        "DirectKV":  {"lat_per_1k": 1.0, "mem_frac": 0.55},  # high savings, moderate latency
        "Pie":       {"lat_per_1k": 0.85, "mem_frac": 0.75},  # low latency, moderate savings
        "Neo":       {"lat_per_1k": 1.1, "mem_frac": 0.80},
        "FlexGen":   {"lat_per_1k": 1.3, "mem_frac": 0.70},
    }

    rows11 = []
    for system, prof in longctx_profiles.items():
        for sl in seq_lengths:
            lat = prof["lat_per_1k"] * (sl / 1024) * (1 + 0.1 * np.log2(sl / 1024))
            lat += rng.normal(0, 0.05)
            lat = max(0.1, lat)
            mem = baseline_mem[sl] * prof["mem_frac"]
            mem_saving = (1 - prof["mem_frac"]) * 100

            rows11.append({
                "system": system, "model": "opt-30b", "seq_length": sl,
                "mean_latency_s": round(lat, 3),
                "p50_latency_s": round(lat * 0.90, 3),
                "p90_latency_s": round(lat * 1.35, 3),
                "p99_latency_s": round(lat * 1.80, 3),
                "gpu_memory_peak_mb": round(mem, 0),
                "memory_saving_pct": round(mem_saving, 1),
                "throughput_tps": round(100 / lat, 0),
            })

    df11 = pd.DataFrame(rows11)
    df11.to_csv(fig11_path, index=False)
    print(f"[sample] Figure 11 data → {fig11_path}")

    return fig10_path, fig11_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Plot offload benchmark results")
    p.add_argument("--fig10-csv", default="results/fig10_latency_vs_rate.csv",
                   help="CSV from bench_offload_serving.py")
    p.add_argument("--fig11-csv", default="results/fig11_longctx.csv",
                   help="CSV from bench_offload_longctx.py")
    p.add_argument("--output-dir", default="results/plots/",
                   help="Directory for output figures")
    p.add_argument("--generate-sample", action="store_true",
                   help="Generate sample data and plot it (for testing)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.generate_sample:
        fig10_csv, fig11_csv = generate_sample_data(args.output_dir)
        args.fig10_csv = fig10_csv
        args.fig11_csv = fig11_csv

    # Figure 10
    if os.path.exists(args.fig10_csv):
        plot_figure10(
            args.fig10_csv,
            os.path.join(args.output_dir, "figure10.pdf"),
        )
    else:
        print(f"[skip] Figure 10: {args.fig10_csv} not found")

    # Figure 11
    if os.path.exists(args.fig11_csv):
        plot_figure11(
            args.fig11_csv,
            os.path.join(args.output_dir, "figure11.pdf"),
        )
    else:
        print(f"[skip] Figure 11: {args.fig11_csv} not found")

    # Figure 12 — crossover analysis
    if os.path.exists(args.fig10_csv) or os.path.exists(args.fig11_csv):
        plot_crossover(
            args.fig10_csv,
            args.fig11_csv,
            os.path.join(args.output_dir, "figure12_crossover.pdf"),
        )


if __name__ == "__main__":
    main()
