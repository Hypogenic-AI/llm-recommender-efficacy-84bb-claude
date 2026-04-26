"""Classical recommender baselines that all rank a fixed candidate pool.

Each baseline implements `rank(pool: dict) -> list[int]` and returns a
permutation of `range(len(pool['candidates']))` representing the ranked order
(index 0 = highest predicted relevance).
"""
from __future__ import annotations

import random
from collections import Counter
from typing import List

import numpy as np

AUDIO_FEATURES = ["danceability", "energy", "valence", "acousticness"]


def random_ranker(pool: dict, seed: int = 0) -> List[int]:
    rng = random.Random(f"{seed}-{pool['user_id']}")
    idx = list(range(len(pool["candidates"])))
    rng.shuffle(idx)
    return idx


def popularity_ranker(pool: dict) -> List[int]:
    """Higher Spotify popularity score = higher rank."""
    cands = pool["candidates"]
    scores = [c["popularity"] for c in cands]
    order = sorted(range(len(cands)), key=lambda i: scores[i], reverse=True)
    return order


def item_knn_ranker(pool: dict, cooc: dict) -> List[int]:
    """Score each candidate by total co-listening with the user's history."""
    cands = pool["candidates"]
    history_sids = [h["spotify_track_id"] for h in pool["history_top"]]
    history_cooc_acc = Counter()
    for sid in history_sids:
        for nbr_sid, c in cooc.get(sid, {}).items():
            history_cooc_acc[nbr_sid] += c
    scores = [history_cooc_acc.get(c["track_id"], 0) for c in cands]
    # tie-break with popularity
    order = sorted(
        range(len(cands)),
        key=lambda i: (scores[i], cands[i]["popularity"]),
        reverse=True,
    )
    return order


def _track_vec(track: dict, all_genres: list, genre_to_idx: dict) -> np.ndarray:
    """Vector representation: one-hot genre + audio features."""
    v = np.zeros(len(all_genres) + len(AUDIO_FEATURES), dtype=np.float32)
    g = track.get("track_genre")
    if g in genre_to_idx:
        v[genre_to_idx[g]] = 1.0
    for j, f in enumerate(AUDIO_FEATURES):
        v[len(all_genres) + j] = float(track.get(f, 0.0))
    return v


def cbf_ranker(pool: dict, catalog) -> List[int]:
    """TF-IDF / cosine over genre tag + audio features.

    Builds a user vector as the mean of the user's history-track vectors,
    then scores each candidate by cosine similarity.
    """
    all_genres = sorted(catalog.df["track_genre"].unique())
    genre_to_idx = {g: i for i, g in enumerate(all_genres)}
    history = pool["history_top"]
    hist_vecs = []
    for h in history:
        key = (h["norm_artist"], h["norm_track"])
        track = catalog.key_to_track.get(key)
        if track is not None:
            hist_vecs.append(_track_vec(track, all_genres, genre_to_idx))
    if not hist_vecs:
        return list(range(len(pool["candidates"])))
    user_vec = np.mean(np.stack(hist_vecs), axis=0)
    user_norm = np.linalg.norm(user_vec) + 1e-9
    cands = pool["candidates"]
    scores = []
    for c in cands:
        v = _track_vec(c, all_genres, genre_to_idx)
        sim = float(np.dot(user_vec, v) / (user_norm * (np.linalg.norm(v) + 1e-9)))
        scores.append(sim)
    order = sorted(range(len(cands)), key=lambda i: scores[i], reverse=True)
    return order


if __name__ == "__main__":
    from src.data_loader import load_catalog, load_lastfm, resolve_to_catalog, per_user_splits
    from src.build_pools import build_global_cooccurrence, build_candidate_pool
    cat = load_catalog()
    lf = load_lastfm()
    lf2 = resolve_to_catalog(lf, cat)
    splits = per_user_splits(lf2, cat)
    cooc = build_global_cooccurrence(lf2)
    uid = list(splits.keys())[0]
    s = splits[uid]
    pool = build_candidate_pool(uid, s["train"], s["test_positives"], cat, cooc)
    print(f"User {uid}, positives at {pool['positive_indices']}")
    print(f"random:  {random_ranker(pool)}")
    print(f"popular: {popularity_ranker(pool)}")
    print(f"item-kNN: {item_knn_ranker(pool, cooc)}")
    print(f"cbf:     {cbf_ranker(pool, cat)}")
