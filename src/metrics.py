"""Evaluation metrics following the Epure et al. (2025) framework.

Each function takes a `pool` (with 'candidates' and 'positive_indices') and a
ranked `order` (permutation of candidate indices), and returns a scalar.
"""
from __future__ import annotations

import math
from typing import List


def recall_at_k(pool: dict, order: List[int], k: int = 10) -> float:
    pos = set(pool["positive_indices"])
    if not pos:
        return float("nan")
    topk = set(order[:k])
    return len(pos & topk) / len(pos)


def ndcg_at_k(pool: dict, order: List[int], k: int = 10) -> float:
    pos = set(pool["positive_indices"])
    if not pos:
        return float("nan")
    dcg = 0.0
    for rank, idx in enumerate(order[:k], 1):
        if idx in pos:
            dcg += 1.0 / math.log2(rank + 1)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(pos), k) + 1))
    return dcg / idcg if idcg > 0 else float("nan")


def mrr(pool: dict, order: List[int]) -> float:
    pos = set(pool["positive_indices"])
    if not pos:
        return float("nan")
    for rank, idx in enumerate(order, 1):
        if idx in pos:
            return 1.0 / rank
    return 0.0


def arp_at_k(pool: dict, order: List[int], k: int = 10) -> float:
    """Average Recommendation Popularity (Abdollahpouri et al.) over top-k."""
    cands = pool["candidates"]
    pops = [cands[i]["popularity"] for i in order[:k]]
    return sum(pops) / len(pops) if pops else float("nan")


def long_tail_share_at_k(
    pool: dict, order: List[int], k: int = 10, threshold: int = 35
) -> float:
    """Fraction of top-k items whose Spotify popularity is below `threshold`.

    Default threshold = 35 = catalog 50th percentile (see EDA).
    """
    cands = pool["candidates"]
    if not order:
        return float("nan")
    n = sum(1 for i in order[:k] if cands[i]["popularity"] < threshold)
    return n / min(k, len(order))


def genre_diversity_at_k(pool: dict, order: List[int], k: int = 10) -> float:
    """Number of distinct genres in top-k, normalized to [0, 1]."""
    cands = pool["candidates"]
    if not order:
        return float("nan")
    genres = set(cands[i]["track_genre"] for i in order[:k])
    return len(genres) / min(k, len(order))


def evaluate_run(pool: dict, order: List[int], k: int = 10) -> dict:
    """Compute all metrics for a single (pool, order) pair."""
    return {
        "recall@10": recall_at_k(pool, order, k),
        "ndcg@10": ndcg_at_k(pool, order, k),
        "mrr": mrr(pool, order),
        "arp@10": arp_at_k(pool, order, k),
        "long_tail_share@10": long_tail_share_at_k(pool, order, k),
        "genre_diversity@10": genre_diversity_at_k(pool, order, k),
    }
