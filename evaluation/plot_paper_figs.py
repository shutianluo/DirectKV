"""
Reproduce paper Figures 10 and 11.

Figure 10: 3-panel, per-token latency (TPOT) vs request rate, 4 systems × 3 models.
Figure 11: 2-panel, (a) TPOT vs seq length (bars), (b) memory saving % vs seq length (bars).

Usage:
  python evaluation/plot_paper_figs.py [--fig10-csv PATH] [--fig11-csv PATH] [--output-dir DIR]
"""

import argparse
import csv
import os
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Style (matching paper)
# ---------------------------------------------------------------------------
COLORS  = {"DirectKV": "#2078b4", "Neo": "#ff7f0f", "Pie": "#2ca02c", "FlexGen": "#d62728"}
MARKERS = {"DirectKV": "o",       "Neo": "x",       "Pie": "s",       "FlexGen": "D"}
LINESZ  = 2.0
MKSZ    = 7

MODEL_LABELS = {
    "llama-3-8b": "LLaMa-3.1-8B",
    "opt-6.7b":   "OPT-8B",          # paper calls it OPT-8B
    "opt-30b":    "OPT-30B",
}

FIG10_MODELS = ["llama-3-8b", "opt-6.7b", "opt-30b"]
SYSTEMS      = ["DirectKV", "Neo", "Pie", "FlexGen"]


# ---------------------------------------------------------------------------
# CSV readers
# ---------------------------------------------------------------------------

def load_fig10(paths):
    """
    Read all fig10 CSVs (any schema with system/model/request_rate/mean_latency).
    Returns dict: (system, model) -> {rate -> tpot_s}
    """
    data = defaultdict(dict)
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        with open(p) as f:
            for row in csv.DictReader(f):
                sys   = row.get("system", "")
                model = row.get("model", "")
                rate  = float(row.get("request_rate", 0))
                # mean_latency: may be TPOT in seconds (Neo, Pie, DirectKV) or other
                tpot  = float(row.get("mean_latency", 0) or 0)
                if sys and model and rate:
                    data[(sys, model)][rate] = tpot
    return data


def load_fig11(paths):
    """
    Read all fig11 CSVs.
    Returns dict: (system, model) -> {seq_len -> {mean_latency_s, memory_saving_pct}}
    """
    data = defaultdict(dict)
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        with open(p) as f:
            for row in csv.DictReader(f):
                sys   = row.get("system", "")
                model = row.get("model", "")
                seq   = int(row.get("seq_length", 0) or 0)
                if not sys or not model or not seq:
                    continue
                lat  = row.get("mean_latency_s", "") or "0"
                msav = row.get("memory_saving_pct", "") or "0"
                try:
                    data[(sys, model)][seq] = {
                        "lat":  float(lat),
                        "msav": float(msav),
                    }
                except ValueError:
                    pass
    return data


# ---------------------------------------------------------------------------
# Figure 10
# ---------------------------------------------------------------------------

def plot_figure10(fig10_data: dict, out_path: Path):
    rates = [5, 10, 15, 20, 25, 30]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=False)

    for col, model in enumerate(FIG10_MODELS):
        ax = axes[col]
        for sys in SYSTEMS:
            d = fig10_data.get((sys, model), {})
            if not d:
                continue
            xs = sorted(d.keys())
            ys = [d[r] for r in xs]
            ax.plot(xs, ys,
                    color=COLORS[sys], marker=MARKERS[sys],
                    linewidth=LINESZ, markersize=MKSZ,
                    label=sys)

        ax.set_xlabel("Request rate (req/s)", fontsize=10)
        if col == 0:
            ax.set_ylabel("Latency (s)", fontsize=10)
        ax.set_title(f"({chr(ord('a')+col)}) {MODEL_LABELS.get(model, model)}", fontsize=10)
        ax.set_xticks([5, 10, 15, 20, 25, 30])
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylim(bottom=0)

    # Legend above all panels
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        # Fallback: try other panels
        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                break
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 1.04))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[fig10] saved → {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 11
# ---------------------------------------------------------------------------

def plot_figure11(fig11_data: dict, model: str, out_path: Path):
    seq_lens = [1024, 2048, 4096, 8192]
    seq_labels = ["1k", "2k", "4k", "8k"]
    x = np.arange(len(seq_lens))
    width = 0.20
    sys_order = ["DirectKV", "Neo", "Pie", "FlexGen"]

    fig, (ax_lat, ax_mem) = plt.subplots(1, 2, figsize=(9, 3.8))

    for i, sys in enumerate(sys_order):
        d = fig11_data.get((sys, model), {})
        lats  = [d.get(s, {}).get("lat",  0) or 0 for s in seq_lens]
        msavs = [d.get(s, {}).get("msav", 0) or 0 for s in seq_lens]

        offset = (i - 1.5) * width
        ax_lat.bar(x + offset, lats,  width, label=sys, color=COLORS[sys],
                   alpha=0.85, edgecolor="black", linewidth=0.5)
        ax_mem.bar(x + offset, msavs, width, label=sys, color=COLORS[sys],
                   alpha=0.85, edgecolor="black", linewidth=0.5)

    for ax, ylabel, subtitle in [
        (ax_lat, "Per-token latency (s)", "(a) Latency"),
        (ax_mem, "GPU Memory Saving (%)", "(b) Memory Saving"),
    ]:
        ax.set_xlabel("Sequence length", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(subtitle, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(seq_labels)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.set_ylim(bottom=0)

    handles, labels = ax_lat.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 1.04))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[fig11] saved → {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir",  default=None)
    ap.add_argument("--fig10-csv",   nargs="+", default=[])
    ap.add_argument("--fig11-csv",   nargs="+", default=[])
    ap.add_argument("--fig11-model", default="opt-30b")
    args = ap.parse_args()

    root = Path(__file__).parent.parent
    out  = Path(args.output_dir) if args.output_dir else root / "results" / "plots"
    out.mkdir(parents=True, exist_ok=True)

    # ---- Fig10 data sources ----
    fig10_paths = args.fig10_csv if args.fig10_csv else []
    # Always include all existing fig10 CSVs (Neo, Pie, FlexGen baselines)
    default_fig10 = [
        root / "results" / "fig10_directkv.csv",
        root / "results" / "fig10a_neo.csv",
        root / "results" / "fig10_pie.csv",
        # OPT baseline placeholders (empty if not yet collected)
    ]
    # If caller provides new DirectKV CG results, prefer those over old ones
    fig10_all = list(fig10_paths) + [str(p) for p in default_fig10]
    fig10_data = load_fig10(fig10_all)

    # If new CG results provided, overwrite old DirectKV entries
    if fig10_paths:
        new_data = load_fig10(fig10_paths)
        for key in list(fig10_data.keys()):
            if key[0] == "DirectKV" and key in new_data:
                fig10_data[key] = new_data[key]

    # ---- Fig11 data sources ----
    fig11_paths = args.fig11_csv if args.fig11_csv else []
    default_fig11 = [
        root / "results" / "fig11_longctx.csv",
        root / "results" / "fig11_neo.csv",
        root / "results" / "fig11_pie.csv",
        root / "results" / "fig11_flexgen.csv",
        root / "results" / "fig11_flexgen_30b.csv",
        root / "results" / "fig11_directkv.csv",
    ]
    fig11_all = list(fig11_paths) + [str(p) for p in default_fig11]
    fig11_data = load_fig11(fig11_all)

    # Apply memory saving calculation for DirectKV OPT-30B if missing
    # saving% = batch×kv / (model_weights + batch×kv) using batch=4, OPT-30B
    OPT30B_WEIGHT_GB = 60.0
    OPT30B_KV_GB = {1024: 1.409, 2048: 2.818, 4096: 5.637, 8192: 11.274}
    BATCH = 4
    directkv_30b = fig11_data.get(("DirectKV", "opt-30b"), {})
    for seq, kv in OPT30B_KV_GB.items():
        entry = directkv_30b.get(seq, {})
        if entry and entry.get("msav", 0) == 0:
            batch_kv = BATCH * kv
            saving = batch_kv / (OPT30B_WEIGHT_GB + batch_kv) * 100
            entry["msav"] = round(saving, 1)
            directkv_30b[seq] = entry

    # ---- Generate figures ----
    plot_figure10(fig10_data, out / "figure10.pdf")
    plot_figure10(fig10_data, out / "figure10.png")
    plot_figure11(fig11_data, args.fig11_model, out / "figure11.pdf")
    plot_figure11(fig11_data, args.fig11_model, out / "figure11.png")

    print(f"\nFigures written to {out}/")


if __name__ == "__main__":
    main()
