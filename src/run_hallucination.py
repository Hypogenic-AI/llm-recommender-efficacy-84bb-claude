"""Experiment 3 — Hallucination diagnostic.

Asks Claude (free-generation mode) to recommend 10 (artist, track) tuples for
each user. Resolves them against the Spotify catalog via the same fuzzy
(artist, track) match used elsewhere. Reports EntitiesResolvedShare = fraction
that resolve.

Outputs:
  results/hallucination.csv   — per-user, per-rec resolution status
  results/hallucination_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.build_pools import build_candidate_pool, build_global_cooccurrence
from src.data_loader import (
    _normalize,
    load_catalog,
    load_lastfm,
    per_user_splits,
    resolve_to_catalog,
)
from src.llm_ranker import CLAUDE_MODEL, GPT_MODEL, llm_free_generate

WORKSPACE = Path(__file__).resolve().parent.parent
RESULTS = WORKSPACE / "results"


def resolve_rec(rec: dict, catalog) -> dict:
    artist = rec.get("artist", "")
    track = rec.get("track", "")
    a = _normalize(artist)
    t = _normalize(track)
    key = (a, t)
    hit = catalog.key_to_track.get(key)
    if hit:
        return {"resolved": True, "match_type": "exact", **hit}
    # Fallback: same artist, fuzzy track substring
    artist_keys = catalog.artist_to_keys.get(a, [])
    for nt, k in artist_keys:
        if t and (t in nt or nt in t):
            hit = catalog.key_to_track[k]
            return {"resolved": True, "match_type": "artist+substr", **hit}
    return {"resolved": False, "match_type": None}


def run() -> None:
    catalog = load_catalog()
    lastfm = load_lastfm()
    lastfm = resolve_to_catalog(lastfm, catalog)
    splits = per_user_splits(lastfm, catalog)
    cooc = build_global_cooccurrence(lastfm)
    print(f"Free-gen on {len(splits)} users.")

    rows = []
    for ui, uid in enumerate(splits):
        s = splits[uid]
        pool = build_candidate_pool(uid, s["train"], s["test_positives"], catalog, cooc)
        for model in (CLAUDE_MODEL, GPT_MODEL):
            try:
                res = llm_free_generate(pool, model=model)
            except Exception as e:
                print(f"  {uid} {model}: {e}")
                continue
            for r in res["recs"]:
                resolved = resolve_rec(r, catalog)
                rows.append({
                    "user_id": uid,
                    "model": model,
                    "artist_pred": r.get("artist", ""),
                    "track_pred": r.get("track", ""),
                    "resolved": resolved["resolved"],
                    "match_type": resolved.get("match_type"),
                    "popularity": resolved.get("popularity"),
                    "track_genre": resolved.get("track_genre"),
                })
            print(f"[{ui+1}/{len(splits)}] {uid} {model}: {len(res['recs'])} recs, "
                  f"{sum(1 for r in res['recs'] if resolve_rec(r, catalog)['resolved'])} resolved")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "hallucination.csv", index=False)
    summary = {}
    for model, sub in df.groupby("model"):
        summary[model] = {
            "n_recs": int(len(sub)),
            "n_resolved": int(sub["resolved"].sum()),
            "entities_resolved_share": float(sub["resolved"].mean()),
            "mean_popularity_resolved": float(sub.loc[sub["resolved"], "popularity"].mean()),
        }
    (RESULTS / "hallucination_summary.json").write_text(json.dumps(summary, indent=2))
    print("Summary:", summary)


if __name__ == "__main__":
    run()
