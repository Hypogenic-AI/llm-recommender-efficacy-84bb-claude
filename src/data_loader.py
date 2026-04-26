"""Load Last.fm listening histories and the Spotify catalog, and build the
per-user train/test split used by all downstream experiments.

Train/test split rule: per user, the LAST 7 days of listens are the test
window; everything before is training. Positives in the test set are tracks
the user listened to in the test window that they did NOT listen to in the
train window (i.e., new-to-user tracks during the test week).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_from_disk

WORKSPACE = Path(__file__).resolve().parent.parent
LASTFM_PATH = WORKSPACE / "datasets/lastfm_1k/lastfm-dataset-50.snappy.parquet"
SPOTIFY_PATH = WORKSPACE / "datasets/spotify_tracks_dataset"


def _normalize(s: str) -> str:
    """Lowercase + strip punctuation/diacritics for fuzzy matching."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"\(.*?\)|\[.*?\]", "", s)  # drop parenthesized qualifiers
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


@dataclass
class Catalog:
    df: pd.DataFrame
    key_to_track: dict  # (norm_artist, norm_track) -> Spotify row dict
    artist_to_keys: dict  # norm_artist -> list of (norm_track, key)


def load_catalog() -> Catalog:
    ds = load_from_disk(str(SPOTIFY_PATH))["train"].to_pandas()
    # Some duplicates because each track may appear under multiple genres.
    # Keep the first occurrence per (artists, track_name) for stable resolution.
    ds["norm_artist"] = ds["artists"].astype(str).map(_normalize)
    ds["norm_track"] = ds["track_name"].astype(str).map(_normalize)
    ds = ds.drop_duplicates(subset=["norm_artist", "norm_track"], keep="first").reset_index(drop=True)
    key_to_track = {}
    artist_to_keys: dict = {}
    for _, row in ds.iterrows():
        key = (row.norm_artist, row.norm_track)
        key_to_track[key] = {
            "track_id": row.track_id,
            "artists": row.artists,
            "track_name": row.track_name,
            "popularity": int(row.popularity),
            "track_genre": row.track_genre,
            "danceability": float(row.danceability),
            "energy": float(row.energy),
            "valence": float(row.valence),
            "acousticness": float(row.acousticness),
            "tempo": float(row.tempo),
        }
        artist_to_keys.setdefault(row.norm_artist, []).append((row.norm_track, key))
    return Catalog(df=ds, key_to_track=key_to_track, artist_to_keys=artist_to_keys)


def load_lastfm() -> pd.DataFrame:
    df = pd.read_parquet(LASTFM_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["norm_artist"] = df["artist_name"].astype(str).map(_normalize)
    df["norm_track"] = df["track_name"].astype(str).map(_normalize)
    return df


def resolve_to_catalog(lastfm: pd.DataFrame, catalog: Catalog) -> pd.DataFrame:
    """Add a `spotify_track_id` (or NaN) column based on (artist, track) match."""
    keys = list(zip(lastfm["norm_artist"], lastfm["norm_track"]))
    sids = [catalog.key_to_track.get(k, {}).get("track_id") for k in keys]
    out = lastfm.copy()
    out["spotify_track_id"] = sids
    return out


def per_user_splits(
    lastfm: pd.DataFrame, catalog: Catalog, test_days: int = 30,
    min_train_listens: int = 50, min_test_positives: int = 2,
) -> dict:
    """Return {user_id: {train: df, test_positives: list[track_dict]}}."""
    splits = {}
    for uid, group in lastfm.groupby("user_id"):
        group = group.sort_values("timestamp")
        max_ts = group["timestamp"].max()
        cutoff = max_ts - pd.Timedelta(days=test_days)
        train = group[group["timestamp"] <= cutoff]
        test = group[group["timestamp"] > cutoff]
        if len(train) < min_train_listens or len(test) == 0:
            continue
        train_keys = set(zip(train["norm_artist"], train["norm_track"]))
        test_pos_keys = []
        seen = set()
        for _, row in test.iterrows():
            key = (row["norm_artist"], row["norm_track"])
            if key in train_keys or key in seen:
                continue
            seen.add(key)
            track = catalog.key_to_track.get(key)
            if track is not None:
                test_pos_keys.append({"key": key, **track})
        if len(test_pos_keys) < min_test_positives:
            continue
        # Restrict train to last ~50 unique tracks for the prompt context.
        # Keep full train for kNN co-occurrence.
        splits[uid] = {
            "train": train,
            "test_positives": test_pos_keys,
        }
    return splits


if __name__ == "__main__":
    cat = load_catalog()
    print(f"Catalog: {len(cat.df):,} unique (artist, track) rows")
    lf = load_lastfm()
    lf2 = resolve_to_catalog(lf, cat)
    print(f"Last.fm: {len(lf2):,} listens; resolved {lf2['spotify_track_id'].notna().sum():,}")
    splits = per_user_splits(lf2, cat)
    print(f"Users with usable splits: {len(splits)}")
    for uid in list(splits.keys())[:3]:
        s = splits[uid]
        print(f"  {uid}: train={len(s['train'])}, test_positives={len(s['test_positives'])}")
