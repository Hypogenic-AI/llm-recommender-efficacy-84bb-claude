"""Main experiment orchestrator.

For every user with a usable train/test split:
  1. Build a single 20-track candidate pool.
  2. Run every method (random, popularity, item-kNN, CBF, Claude with history,
     GPT-4.1 with history, Claude no-history) to produce a ranking.
  3. Compute all metrics and append to a results table.

Outputs:
  results/per_user_metrics.csv  — one row per (user, method, run)
  results/llm_runs.json         — raw LLM responses (for inspection)
  results/config.json           — environment + hyperparameters
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines import (
    cbf_ranker,
    item_knn_ranker,
    popularity_ranker,
    random_ranker,
)
from src.build_pools import build_candidate_pool, build_global_cooccurrence
from src.data_loader import load_catalog, load_lastfm, per_user_splits, resolve_to_catalog
from src.llm_ranker import CLAUDE_MODEL, GPT_MODEL, llm_rank
from src.metrics import evaluate_run

WORKSPACE = Path(__file__).resolve().parent.parent
RESULTS = WORKSPACE / "results"
LOGS = WORKSPACE / "logs"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def run(args) -> None:
    set_seed(args.seed)
    RESULTS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)

    print("Loading data ...")
    catalog = load_catalog()
    lastfm = load_lastfm()
    lastfm = resolve_to_catalog(lastfm, catalog)
    splits = per_user_splits(lastfm, catalog, test_days=args.test_days)
    print(f"  Catalog: {len(catalog.df):,}")
    print(f"  Last.fm rows: {len(lastfm):,}")
    print(f"  Users with usable splits: {len(splits)}")

    print("Building global co-occurrence ...")
    cooc = build_global_cooccurrence(lastfm)

    config = {
        "seed": args.seed,
        "test_days": args.test_days,
        "n_users": len(splits),
        "claude_model": CLAUDE_MODEL,
        "gpt_model": GPT_MODEL,
        "n_llm_runs": args.n_llm_runs,
        "max_users": args.max_users,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    (RESULTS / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Config written to {RESULTS / 'config.json'}")

    methods = [
        ("random", lambda pool: random_ranker(pool, seed=args.seed)),
        ("popularity", popularity_ranker),
        ("item_knn", lambda pool: item_knn_ranker(pool, cooc)),
        ("cbf", lambda pool: cbf_ranker(pool, catalog)),
    ]

    rows = []
    llm_run_records = []
    user_ids = list(splits.keys())[: args.max_users] if args.max_users else list(splits.keys())
    print(f"Running on {len(user_ids)} users.")

    for ui, uid in enumerate(user_ids):
        s = splits[uid]
        try:
            pool = build_candidate_pool(uid, s["train"], s["test_positives"], catalog, cooc)
        except Exception as e:
            print(f"[{ui}] {uid}: pool build failed: {e}")
            continue
        n_pos = len(pool["positive_indices"])
        if n_pos == 0:
            print(f"[{ui}] {uid}: 0 positives in pool — skipping")
            continue
        print(f"[{ui+1}/{len(user_ids)}] {uid}  positives={pool['positive_indices']}  ", end="", flush=True)

        # 1. Classical baselines (deterministic — single run each)
        for name, fn in methods:
            order = fn(pool)
            m = evaluate_run(pool, order)
            rows.append({"user_id": uid, "method": name, "run": 0, **m})

        # 2. LLM arms — n_llm_runs each
        llm_specs = [
            ("claude_with_history", CLAUDE_MODEL, True),
            ("gpt41_with_history", GPT_MODEL, True),
            ("claude_no_history", CLAUDE_MODEL, False),
        ]
        for arm_name, model, use_history in llm_specs:
            for run_i in range(args.n_llm_runs if use_history else 1):
                try:
                    res = llm_rank(pool, model=model, use_history=use_history)
                except Exception as e:
                    print(f"\n   {arm_name} run {run_i} failed: {e}")
                    continue
                m = evaluate_run(pool, res["order"])
                rows.append({
                    "user_id": uid, "method": arm_name, "run": run_i,
                    **m,
                    "tokens_in": res["tokens_in"], "tokens_out": res["tokens_out"],
                    "latency_s": res["latency_s"], "parsed_ok": res["parsed_ok"],
                })
                llm_run_records.append({
                    "user_id": uid, "method": arm_name, "run": run_i,
                    "model": model,
                    "order": res["order"], "raw": res["raw"],
                    "positive_indices": pool["positive_indices"],
                    "parsed_ok": res["parsed_ok"],
                    "tokens_in": res["tokens_in"], "tokens_out": res["tokens_out"],
                    "latency_s": res["latency_s"],
                })
        # Periodic save
        if (ui + 1) % 5 == 0 or ui == len(user_ids) - 1:
            pd.DataFrame(rows).to_csv(RESULTS / "per_user_metrics.csv", index=False)
            (RESULTS / "llm_runs.json").write_text(json.dumps(llm_run_records, indent=2))
        print("done")

    pd.DataFrame(rows).to_csv(RESULTS / "per_user_metrics.csv", index=False)
    (RESULTS / "llm_runs.json").write_text(json.dumps(llm_run_records, indent=2))
    print(f"\nSaved {len(rows)} metric rows and {len(llm_run_records)} LLM runs.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_days", type=int, default=30)
    p.add_argument("--n_llm_runs", type=int, default=3)
    p.add_argument("--max_users", type=int, default=None)
    args = p.parse_args()
    run(args)
