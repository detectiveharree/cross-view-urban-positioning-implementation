#!/usr/bin/env python3
"""
analyse_results.py — Analyse positioning accuracy broken down by environment type.

Joins:
    results/results_detail.json      — from run_positioning.py
    results/environment_labels.json  — from classify_environment.py
    results/coverage.csv             — from segment_images.py (optional)

Outputs:
    Console: per-environment metrics table
    results/plots/accuracy_by_environment.png — bar charts
    results/plots/coverage_by_environment.png — class coverage breakdown (if available)
    results/environment_analysis.csv          — full joined table for custom analysis

Usage:
    python analyse_results.py
    python analyse_results.py --mode depth_weighted --regime known
    python analyse_results.py --results results/my_run/results_detail.json
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

TOP_N     = [5, 10, 15, 20]
CLASS_NAMES = ["building", "lamp_post", "tree", "stop_sign",
               "fire_hydrant", "road", "sidewalk", "crosswalk"]


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_results(path) -> pd.DataFrame:
    with open(path) as f:
        detail = json.load(f)
    rows = []
    for mode, regimes in detail.items():
        for regime, samples in regimes.items():
            for s in samples:
                rows.append({
                    "image_id":          str(s["image_id"]),
                    "mode":              mode,
                    "regime":            regime,
                    "bracket_percentile": s["bracket_percentile"],
                    "error_m":           s["error_m"],
                    "algo_time_s":       s.get("algo_time_s", 0),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame) -> dict:
    pcts = df["bracket_percentile"]
    errs = df["error_m"]
    m    = {"n": len(df), "median_error_m": errs.median(), "mean_error_m": errs.mean()}
    for t in TOP_N:
        m[f"top_{t}_pct"] = (pcts <= t).mean() * 100
    return m


def print_table(metrics_by_env: dict, title: str = "Results by Environment"):
    print(f"\n{'='*80}")
    print(title)
    print(f"{'='*80}")
    envs  = list(metrics_by_env.keys())
    cw    = max(12, max(len(e) for e in envs) + 2)
    label_w = 28

    header = f"{'Metric':<{label_w}}" + "".join(f"{e:<{cw}}" for e in envs)
    print(header)
    print("-" * len(header))

    rows = [("Samples", "n", "d"), ("Median error (m)", "median_error_m", ".1f"),
            ("Mean error (m)", "mean_error_m", ".1f")]
    for t in TOP_N:
        rows.append((f"Top-{t}% inclusion (%)", f"top_{t}_pct", ".1f"))

    for label, key, fmt in rows:
        line = f"{label:<{label_w}}"
        for e in envs:
            val = metrics_by_env[e].get(key, float("nan"))
            if fmt == "d":
                line += f"{int(val):<{cw}}"
            else:
                line += f"{val:<{cw}{fmt}}"
        print(line)
    print("=" * len(header))


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------

def plot_accuracy(metrics_by_env: dict, output_path: str):
    envs   = list(metrics_by_env.keys())
    n_envs = len(envs)
    cols   = min(n_envs, 5)
    fig, axes = plt.subplots(1, 2, figsize=(max(10, cols * 2.5), 5))

    # Top-10% inclusion
    ax  = axes[0]
    vals = [metrics_by_env[e]["top_10_pct"] for e in envs]
    bars = ax.bar(envs, vals, color="steelblue", edgecolor="white", width=0.6)
    ax.set_ylabel("Top-10% Inclusion Rate (%)")
    ax.set_title("Positioning Accuracy by Environment\n(higher = better)")
    ax.set_ylim(0, 105)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # Median error
    ax  = axes[1]
    vals = [metrics_by_env[e]["median_error_m"] for e in envs]
    bars = ax.bar(envs, vals, color="coral", edgecolor="white", width=0.6)
    ax.set_ylabel("Median Positioning Error (m)")
    ax.set_title("Median Error by Environment\n(lower = better)")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f} m", ha="center", va="bottom", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_coverage(df_joined: pd.DataFrame, output_path: str):
    available = [c for c in CLASS_NAMES if c in df_joined.columns]
    if not available:
        return

    envs = sorted(df_joined["label"].dropna().unique())
    fig, ax = plt.subplots(figsize=(max(10, len(envs) * 2.5), 5))

    x    = np.arange(len(envs))
    w    = 0.75 / len(available)
    cmap = plt.cm.get_cmap("tab10", len(available))

    for i, cls in enumerate(available):
        vals = [df_joined[df_joined["label"] == e][cls].mean() * 100
                for e in envs]
        ax.bar(x + i*w - 0.375 + w/2, vals, width=w,
               label=cls.replace("_", " "), color=cmap(i), edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(envs, rotation=30, ha="right")
    ax.set_ylabel("Mean Pixel Coverage (%)")
    ax.set_title("YOLO Class Coverage by Environment\n"
                 "(classes in bold are used in positioning)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    # Bold labels for classes used in positioning
    used = {"building", "road", "sidewalk", "crosswalk"}
    for lbl in ax.get_legend().get_texts():
        if lbl.get_text().replace(" ", "_") in used:
            lbl.set_fontweight("bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_inclusion_curves(metrics_by_env: dict, output_path: str):
    """Top-N% inclusion for N in [5, 10, 15, 20] per environment."""
    envs = [e for e in metrics_by_env if e != "ALL"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for env in envs:
        m = metrics_by_env[env]
        vals = [m[f"top_{t}_pct"] for t in TOP_N]
        ax.plot(TOP_N, vals, marker="o", label=env)
    ax.set_xlabel("Top-N threshold (%)")
    ax.set_ylabel("Inclusion rate (%)")
    ax.set_title("Top-N% Inclusion Curves by Environment")
    ax.set_xticks(TOP_N)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",    default="results/results_detail.json")
    parser.add_argument("--env-labels", default="results/environment_labels.json")
    parser.add_argument("--coverage",   default="results/coverage.csv")
    parser.add_argument("--mode",       default="depth_weighted",
                        help="Algorithm mode to analyse (depth_weighted or planarity_occlusion)")
    parser.add_argument("--regime",     default="known",
                        help="Heading regime to analyse (known, window_5deg, free, etc.)")
    parser.add_argument("--min-samples", type=int, default=5,
                        help="Minimum images per environment to include in analysis")
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"No results at {args.results}. Run run_positioning.py first.")
        return
    if not os.path.exists(args.env_labels):
        print(f"No environment labels at {args.env_labels}. Run classify_environment.py first.")
        return

    df = load_results(args.results)
    df = df[(df["mode"] == args.mode) & (df["regime"] == args.regime)].copy()
    print(f"Loaded {len(df)} samples (mode={args.mode}, regime={args.regime})")

    with open(args.env_labels) as f:
        env_labels = json.load(f)

    df["label"] = df["image_id"].map(lambda x: env_labels.get(x, {}).get("label"))
    matched = df["label"].notna().sum()
    print(f"  {matched}/{len(df)} images matched to environment labels")

    df_known = df[df["label"].notna()].copy()

    # Compute per-environment metrics
    metrics_by_env = {}
    for env in sorted(df_known["label"].unique()):
        grp = df_known[df_known["label"] == env]
        if len(grp) >= args.min_samples:
            metrics_by_env[env] = compute_metrics(grp)

    metrics_by_env["ALL"] = compute_metrics(df_known)

    print_table(metrics_by_env)

    os.makedirs("results/plots", exist_ok=True)
    plot_accuracy(metrics_by_env, "results/plots/accuracy_by_environment.png")
    plot_inclusion_curves(metrics_by_env, "results/plots/inclusion_curves.png")

    # Optional: join with coverage stats
    df_out = df_known.copy()
    if os.path.exists(args.coverage):
        df_cov = pd.read_csv(args.coverage)
        df_cov["image_id"] = df_cov["image_id"].astype(str)
        df_out = df_out.merge(df_cov, on="image_id", how="left")
        print(f"\nJoined with coverage data ({df_cov['image_id'].isin(df_out['image_id']).sum()} matches)")
        plot_coverage(df_out, "results/plots/coverage_by_environment.png")

        # Correlation: does coverage of positioning classes correlate with accuracy?
        print("\nSpearman correlation: class coverage vs bracket_percentile (lower = better localisation)")
        for cls in ["building", "road", "sidewalk", "crosswalk", "total_detected"]:
            if cls in df_out.columns:
                corr = df_out[["bracket_percentile", cls]].dropna().corr(method="spearman")
                print(f"  {cls:20s}: {corr.loc['bracket_percentile', cls]:+.3f}")

    df_out.to_csv("results/environment_analysis.csv", index=False)
    print(f"\nFull joined table saved to results/environment_analysis.csv")
    print("Open this in Excel, R, or pandas for further analysis.")


if __name__ == "__main__":
    main()
