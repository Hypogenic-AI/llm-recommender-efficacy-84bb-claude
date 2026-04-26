"""Build a 20-track candidate pool per user that mixes (positives + item-kNN
candidates + popularity candidates).

The same pool is used by every method so that differences in ranking are
attributable to the ranking decision and not to candidate generation.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd

POOL_SIZE = 20
N_POSITIVES_CAP = 5
HISTORY_CONTEXT_TRACKS = 50


def build_global_cooccurrence(lastfm: pd.DataFrame) -> dict:
    """Item-item co-listening counts.

    Two tracks co-occur if they appear in the same user's history. We use the
    *resolved* (Spotify-matched) tracks only, since unresolved tracks aren't
    in any candidate pool.
    """
    cooc = defaultdict(Counter)
    df = lastfm[lastfm["spotify_track_id"].notna()][["user_id", "spotify_track_id"]].drop_duplicates()
    user_to_tracks = df.groupby("user_id")["spotify_track_id"].apply(set).to_dict()
    for tracks in user_to_tracks.values():
        ts = list(tracks)
        for i, a in enumerate(ts):
            for b in ts:
                if a != b:
                    cooc[a][b] += 1
    return cooc


def get_user_top_tracks(train: pd.DataFrame, n: int = HISTORY_CONTEXT_TRACKS) -> List[dict]:
    """Top-N most listened-to (resolved) tracks for the user, by play count.

    Used as the user history context shown to all methods.
    """
    resolved = train[train["spotify_track_id"].notna()]
    counts = resolved.groupby(
        ["spotify_track_id", "artist_name", "track_name", "norm_artist", "norm_track"]
    ).size().reset_index(name="plays")
    counts = counts.sort_values("plays", ascending=False).head(n)
    return counts.to_dict("records")


def build_candidate_pool(
    user_id: str,
    train: pd.DataFrame,
    test_positives: List[dict],
    catalog,  # Catalog dataclass
    cooc: dict,
    pool_size: int = POOL_SIZE,
    seed: int = 0,
) -> dict:
    """Build a deduplicated 20-track candidate pool with positives mixed in.

    Returns a dict with:
      candidates: list of catalog track dicts (length = pool_size)
      positive_indices: indices of held-out positives in `candidates`
      seed: random seed used for shuffling
    """
    # Derive a per-user seed so positives don't always land at the same index.
    rng = random.Random(f"{seed}-{user_id}")
    chosen_keys = set()
    chosen = []

    # Resolve user history keys to exclude from candidates.
    user_train_keys = set(zip(train["norm_artist"], train["norm_track"]))

    # 1. Up to N_POSITIVES_CAP positives.
    positives = test_positives[:N_POSITIVES_CAP]
    pos_indices_planned = []
    for p in positives:
        key = p["key"]
        if key in chosen_keys:
            continue
        chosen_keys.add(key)
        chosen.append(p)
        pos_indices_planned.append(len(chosen) - 1)

    # 2. Item-kNN candidates: tracks most co-listened with the user's recent
    # history (top 50 by play-count).
    history_top = get_user_top_tracks(train, n=HISTORY_CONTEXT_TRACKS)
    score = Counter()
    for h in history_top:
        sid = h["spotify_track_id"]
        for nbr_sid, c in cooc.get(sid, {}).items():
            score[nbr_sid] += c
    knn_track_ids = [tid for tid, _ in score.most_common(200)]

    # 3. Build a sid -> catalog dict via reverse lookup.
    sid_to_track = {v["track_id"]: v for v in catalog.key_to_track.values()}

    # Top user genres for popularity stratification.
    top_genres = Counter()
    for h in history_top:
        key = (h["norm_artist"], h["norm_track"])
        track = catalog.key_to_track.get(key)
        if track:
            top_genres[track["track_genre"]] += 1
    user_genres = [g for g, _ in top_genres.most_common(5)]
    if not user_genres:
        user_genres = list(catalog.df["track_genre"].unique()[:5])

    # Add kNN candidates not already in user history.
    for tid in knn_track_ids:
        track = sid_to_track.get(tid)
        if track is None:
            continue
        norm_key = None
        # find normalized key for this track
        for k, v in catalog.key_to_track.items():
            if v["track_id"] == tid:
                norm_key = k
                break
        if norm_key is None or norm_key in user_train_keys or norm_key in chosen_keys:
            continue
        chosen_keys.add(norm_key)
        chosen.append({"key": norm_key, **track})
        if len(chosen) >= pool_size:
            break

    # 4. Popularity candidates: top tracks within user's top genres, not in
    # history or already chosen.
    if len(chosen) < pool_size:
        pop_pool = catalog.df[catalog.df["track_genre"].isin(user_genres)].sort_values(
            "popularity", ascending=False
        )
        for _, row in pop_pool.iterrows():
            key = (row.norm_artist, row.norm_track)
            if key in user_train_keys or key in chosen_keys:
                continue
            track = catalog.key_to_track[key]
            chosen_keys.add(key)
            chosen.append({"key": key, **track})
            if len(chosen) >= pool_size:
                break

    # 5. Final shuffle (seeded for reproducibility) and re-find positive indices.
    rng.shuffle(chosen)
    pos_keys = {tuple(p["key"]) for p in positives}
    positive_indices = [i for i, c in enumerate(chosen) if tuple(c["key"]) in pos_keys]
    return {
        "user_id": user_id,
        "candidates": chosen,
        "positive_indices": positive_indices,
        "history_top": history_top,
        "user_genres": user_genres,
        "seed": seed,
    }


if __name__ == "__main__":
    from src.data_loader import load_catalog, load_lastfm, resolve_to_catalog, per_user_splits
    cat = load_catalog()
    lf = load_lastfm()
    lf2 = resolve_to_catalog(lf, cat)
    splits = per_user_splits(lf2, cat)
    cooc = build_global_cooccurrence(lf2)
    print(f"Co-occurrence built for {len(cooc):,} tracks")

    for uid in list(splits.keys())[:3]:
        s = splits[uid]
        pool = build_candidate_pool(uid, s["train"], s["test_positives"], cat, cooc)
        print(f"{uid}: pool_size={len(pool['candidates'])}, positives at indices {pool['positive_indices']}")
        print(f"  user_genres: {pool['user_genres']}")
