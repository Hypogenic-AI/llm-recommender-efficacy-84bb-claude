"""Personalization-gain Wilcoxon, run-to-run variance at T=0, and cost
accounting. Saved separately because they touch raw run-level data, not the
per-user aggregate."""
import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("results/per_user_metrics.csv")

# Personalization gain: Claude with vs without history
piv = df.groupby(["user_id", "method"])[["recall@10", "ndcg@10", "mrr"]].median().reset_index()
w = piv[piv.method == "claude_with_history"].set_index("user_id")
n = piv[piv.method == "claude_no_history"].set_index("user_id")
common = w.index.intersection(n.index)
print("Personalization gain (claude_with_history - claude_no_history):")
gain_rows = []
for m in ["recall@10", "ndcg@10", "mrr"]:
    diff = w.loc[common, m] - n.loc[common, m]
    stat, p = stats.wilcoxon(diff)
    d = diff.mean() / (diff.std(ddof=1) + 1e-9)
    gain_rows.append({
        "metric": m, "mean_diff": diff.mean(), "median_diff": diff.median(),
        "p_value": p, "cohens_d": d, "n_pairs": len(diff),
    })
    print(f"  {m}: mean_diff={diff.mean():+.3f}, median_diff={diff.median():+.3f}, p={p:.4f}, cohens_d={d:.2f}")
pd.DataFrame(gain_rows).to_csv("results/personalization_gain.csv", index=False)
print()

# Run-to-run variance at T=0
llm = df[df.method.isin(["claude_with_history", "gpt41_with_history"]) & (df["parsed_ok"] == True)]
print("Per-user variance across 3 runs at T=0.0 (NDCG@10):")
v = llm.groupby(["method", "user_id"])["ndcg@10"].std().reset_index()
print(v.groupby("method")["ndcg@10"].agg(["mean", "median", "max"]).round(3))
v.to_csv("results/run_variance.csv", index=False)
print()

# Cost estimate (OpenRouter approx pricing as of 2026-04)
llm_cost = df[df.method.isin(["claude_with_history", "gpt41_with_history", "claude_no_history"])].copy()
prices = {
    "claude_with_history": (3.0, 15.0),
    "gpt41_with_history": (2.0, 8.0),
    "claude_no_history": (3.0, 15.0),
}
def cost(row):
    pi, po = prices[row.method]
    return (row.tokens_in * pi + row.tokens_out * po) / 1e6
llm_cost["cost_usd"] = llm_cost.apply(cost, axis=1)
print("Total spent by method:")
print(llm_cost.groupby("method")["cost_usd"].sum().round(4))
total = llm_cost.cost_usd.sum()
print(f"Total: ${total:.3f}")
llm_cost.groupby("method")["cost_usd"].sum().to_csv("results/cost_per_method.csv")

# Hallucination popularity
import json
hf = json.load(open("results/hallucination_summary.json"))
print("\nHallucination summary:")
for m, s in hf.items():
    print(f"  {m}: resolved={s['entities_resolved_share']:.2%}, "
          f"mean popularity of resolved={s['mean_popularity_resolved']:.1f}")
