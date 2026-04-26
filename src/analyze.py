"""Statistical analysis and visualization of experiment results.

Reads:
  results/per_user_metrics.csv  — per-user, per-method metrics (full run)
  results/hallucination_summary.json
Writes:
  results/summary_stats.csv      — mean, median, std, 95% CI per method
  results/wilcoxon_tests.csv     — paired Wilcoxon: each LLM arm vs each baseline
  figures/*.png                  — bar/box plots
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

WORKSPACE = Path(__file__).resolve().parent.parent
RESULTS = WORKSPACE / "results"
FIGURES = WORKSPACE / "figures"

METRICS = [
    "recall@10",
    "ndcg@10",
    "mrr",
    "arp@10",
    "long_tail_share@10",
    "genre_diversity@10",
]
LLM_ARMS = ["claude_with_history", "gpt41_with_history", "claude_no_history"]
BASELINES = ["random", "popularity", "item_knn", "cbf"]
ALL_METHODS = BASELINES + LLM_ARMS

# Display order — best baseline competitors first
PLOT_ORDER = ["random", "popularity", "item_knn", "cbf", "claude_no_history",
              "gpt41_with_history", "claude_with_history"]


def aggregate_per_user(df: pd.DataFrame) -> pd.DataFrame:
    """For LLM arms with multiple runs, take median across runs (per Hou et al.)."""
    agg = df.groupby(["user_id", "method"], as_index=False)[METRICS].median()
    return agg


def summary_stats(per_user: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, sub in per_user.groupby("method"):
        for metric in METRICS:
            vals = sub[metric].dropna().values
            if len(vals) == 0:
                continue
            ci = stats.bootstrap(
                (vals,), np.mean, n_resamples=2000, confidence_level=0.95,
                random_state=42,
            ).confidence_interval
            rows.append({
                "method": method,
                "metric": metric,
                "n": len(vals),
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "ci_low": float(ci.low),
                "ci_high": float(ci.high),
            })
    return pd.DataFrame(rows)


def wilcoxon_tests(per_user: pd.DataFrame) -> pd.DataFrame:
    """Paired Wilcoxon for each (LLM arm, baseline, metric) triple."""
    rows = []
    pivot = per_user.pivot(index="user_id", columns="method", values=METRICS)
    # pivot is multiindexed columns: (metric, method)
    for arm in LLM_ARMS:
        for base in BASELINES:
            for metric in METRICS:
                if (metric, arm) not in pivot.columns or (metric, base) not in pivot.columns:
                    continue
                a = pivot[(metric, arm)]
                b = pivot[(metric, base)]
                pair = pd.concat([a, b], axis=1).dropna()
                if len(pair) < 5:
                    continue
                diff = pair.iloc[:, 0] - pair.iloc[:, 1]
                if (diff == 0).all():
                    p = 1.0
                    stat = 0.0
                else:
                    try:
                        stat, p = stats.wilcoxon(diff, zero_method="wilcox", correction=False)
                    except ValueError:
                        stat, p = float("nan"), float("nan")
                # Cohen's d for paired data (effect size)
                d = float(diff.mean() / (diff.std(ddof=1) + 1e-9))
                rows.append({
                    "llm_arm": arm,
                    "baseline": base,
                    "metric": metric,
                    "n_pairs": len(pair),
                    "mean_diff": float(diff.mean()),
                    "median_diff": float(diff.median()),
                    "wilcoxon_stat": float(stat) if stat is not None else float("nan"),
                    "p_value": float(p),
                    "cohens_d": d,
                    "llm_better_at_alpha_0.05": (p < 0.05 and diff.mean() > 0),
                })
    return pd.DataFrame(rows)


def plot_metric_bars(summary: pd.DataFrame, metric: str, ylabel: str, out: Path) -> None:
    sub = summary[summary["metric"] == metric].set_index("method").reindex(PLOT_ORDER)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    means = sub["mean"].values
    yerr = np.stack([
        sub["mean"].values - sub["ci_low"].values,
        sub["ci_high"].values - sub["mean"].values,
    ])
    bar_colors = []
    for m in PLOT_ORDER:
        if "claude" in m and "with_history" in m:
            bar_colors.append("#d62728")  # red
        elif "gpt" in m:
            bar_colors.append("#ff7f0e")  # orange
        elif "claude" in m and "no_history" in m:
            bar_colors.append("#9467bd")  # purple
        else:
            bar_colors.append("#1f77b4")  # blue
    ax.bar(range(len(sub)), means, yerr=yerr, color=bar_colors, capsize=4, alpha=0.85)
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(PLOT_ORDER, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{metric} (mean ± 95% bootstrap CI, N={int(sub['n'].iloc[0])} users)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()


def plot_metric_box(per_user: pd.DataFrame, metric: str, ylabel: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = []
    labels = []
    for m in PLOT_ORDER:
        sub = per_user[per_user["method"] == m][metric].dropna().values
        if len(sub):
            data.append(sub)
            labels.append(m)
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, m in zip(bp["boxes"], labels):
        if "claude_with_history" in m:
            patch.set_facecolor("#d62728")
        elif "gpt" in m:
            patch.set_facecolor("#ff7f0e")
        elif "claude_no_history" in m:
            patch.set_facecolor("#9467bd")
        else:
            patch.set_facecolor("#1f77b4")
        patch.set_alpha(0.65)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{metric} per-user distribution")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(RESULTS / "per_user_metrics.csv")
    print(f"Loaded {len(df)} rows; methods: {df['method'].unique().tolist()}")
    per_user = aggregate_per_user(df)
    summary = summary_stats(per_user)
    summary.to_csv(RESULTS / "summary_stats.csv", index=False)
    print(f"Wrote summary_stats.csv: {len(summary)} rows")

    wilcox = wilcoxon_tests(per_user)
    wilcox.to_csv(RESULTS / "wilcoxon_tests.csv", index=False)
    print(f"Wrote wilcoxon_tests.csv: {len(wilcox)} rows")

    # Pretty-print summary tables
    print("\n=== SUMMARY (mean per method) ===")
    pivot = per_user.groupby("method")[METRICS].mean().reindex(PLOT_ORDER).round(3)
    print(pivot.to_string())

    print("\n=== Headline Wilcoxon: claude_with_history vs CF baselines (recall@10, ndcg@10) ===")
    head = wilcox[
        (wilcox["llm_arm"] == "claude_with_history")
        & (wilcox["metric"].isin(["recall@10", "ndcg@10"]))
    ][["baseline", "metric", "n_pairs", "mean_diff", "p_value", "cohens_d"]]
    print(head.round(3).to_string(index=False))

    # Plots
    plot_metric_bars(summary, "recall@10", "Recall@10", FIGURES / "recall_at_10.png")
    plot_metric_bars(summary, "ndcg@10", "NDCG@10", FIGURES / "ndcg_at_10.png")
    plot_metric_bars(summary, "arp@10", "Avg Recommendation Popularity", FIGURES / "arp_at_10.png")
    plot_metric_bars(summary, "long_tail_share@10", "Long-tail share", FIGURES / "long_tail_share.png")
    plot_metric_bars(summary, "genre_diversity@10", "Genre Diversity@10", FIGURES / "genre_diversity.png")
    plot_metric_box(per_user, "recall@10", "Recall@10", FIGURES / "recall_box.png")
    plot_metric_box(per_user, "ndcg@10", "NDCG@10", FIGURES / "ndcg_box.png")

    # Cost / latency
    cost = df[df["latency_s"].notna()].groupby("method").agg(
        mean_latency_s=("latency_s", "mean"),
        median_latency_s=("latency_s", "median"),
        mean_tokens_in=("tokens_in", "mean"),
        mean_tokens_out=("tokens_out", "mean"),
    ).round(2)
    cost.to_csv(RESULTS / "cost_latency.csv")
    print("\n=== Cost / latency ===")
    print(cost.to_string())

    # Hallucination summary (if available)
    hf = RESULTS / "hallucination_summary.json"
    if hf.exists():
        print("\n=== Hallucination ===")
        print(hf.read_text())

    print(f"\nFigures saved to {FIGURES}/")


if __name__ == "__main__":
    main()
