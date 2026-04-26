# Datasets

This directory holds the datasets used to test whether LLMs can match or beat
Spotify-style recommendations when given exported listening history. Large data
files are excluded from git (see `.gitignore`); follow the download instructions
below to recreate the local copy.

## Why these datasets

The hypothesis requires three things:
1. A **user listening history** that resembles a Spotify export (user → tracks
   listened, with timestamps).
2. A **track catalog** with rich metadata so the LLM has something to retrieve
   from when it generates recommendations.
3. **Held-out interactions** so we can score recommendations against what users
   actually listened to next.

Spotify's official Million Playlist Dataset is no longer publicly downloadable,
so we use the Last.fm 1K dataset as a proxy listening history (timestamped
plays per user) and the maharshipandya Spotify Tracks dataset as the track
catalog (Spotify track IDs + audio features + popularity).

## Dataset 1: Spotify Tracks Dataset (HuggingFace)

### Overview
- **Source**: HuggingFace `maharshipandya/spotify-tracks-dataset`
- **Size**: 114,000 tracks, ~25 MB on disk
- **Format**: HuggingFace `Dataset` (saved via `save_to_disk`)
- **Task**: Item catalog with audio features for content-based baselines
- **Splits**: single `train` split
- **License**: BSD

### Columns
`track_id, artists, album_name, track_name, popularity, duration_ms, explicit,
danceability, energy, key, loudness, mode, speechiness, acousticness,
instrumentalness, liveness, valence, tempo, time_signature, track_genre`

### Download Instructions

```python
from datasets import load_dataset
ds = load_dataset("maharshipandya/spotify-tracks-dataset")
ds.save_to_disk("datasets/spotify_tracks_dataset")
```

### Loading

```python
from datasets import load_from_disk
ds = load_from_disk("datasets/spotify_tracks_dataset")
df = ds["train"].to_pandas()
```

A 10-record sample is in `spotify_tracks_dataset/samples.json`.

### Notes
- 114 distinct `track_genre` values (~1000 tracks per genre).
- `popularity` is Spotify's 0-100 popularity score, useful for popularity-bias
  diagnostics (Sguerra et al. 2025; Hou et al. 2023 both call out popularity bias
  in LLM recommendations).
- Audio features (danceability, energy, valence, etc.) enable a content-based
  filtering baseline analogous to the TF-IDF/cosine baseline in Boadana et al.
  2025 (`papers/2508.11671`).

## Dataset 2: Last.fm 1K Users — 50-user sample

### Overview
- **Source**: GitHub eifuentes/lastfm-dataset-1K (release v1.0)
- **Size**: 50 users, ~776K listening events, ~25 MB on disk
- **Format**: snappy-compressed Parquet
- **Task**: Per-user time-stamped listening history (proxy for Spotify export)
- **Period**: 2005-02-14 to 2009-06-19
- **License**: Non-commercial research use (Last.fm permission required for
  redistribution)

### Columns
`user_id, timestamp, artist_id, artist_name, track_id, track_name`

### Download Instructions

```bash
mkdir -p datasets/lastfm_1k
curl -L -o datasets/lastfm_1k/lastfm-dataset-50.snappy.parquet \
  https://github.com/eifuentes/lastfm-dataset-1K/releases/download/v1.0/lastfm-dataset-50.snappy.parquet
curl -L -o datasets/lastfm_1k/userid-profile.tsv.zip \
  https://github.com/eifuentes/lastfm-dataset-1K/releases/download/v1.0/userid-profile.tsv.zip
curl -L -o datasets/lastfm_1k/README.txt \
  https://github.com/eifuentes/lastfm-dataset-1K/releases/download/v1.0/README.txt

# Extract user profile demographics
python -c "import zipfile; zipfile.ZipFile('datasets/lastfm_1k/userid-profile.tsv.zip').extractall('datasets/lastfm_1k/')"
```

### Full 1K-user dataset (optional, larger)
If you need all 992 users, swap the URL for `lastfm-dataset-1k.snappy.parquet`
(~877 MB, ~19M events).

### Loading

```python
import pandas as pd
df = pd.read_parquet("datasets/lastfm_1k/lastfm-dataset-50.snappy.parquet")
profiles = pd.read_csv("datasets/lastfm_1k/userid-profile.tsv", sep="\t")
```

A 20-row sample is in `lastfm_1k/samples.csv`.

### Notes
- 50 users means experiments are tractable in minutes; if statistical power is
  needed, drop in the 1K-user file (same schema).
- Timestamps allow chronological train/test split (e.g., last 30 days held out
  for evaluation), mirroring the protocol in Sguerra et al. 2025.
- Track-level data (no Spotify track IDs); join to the Spotify Tracks Dataset
  by `(artist_name, track_name)` is fuzzy. For experiments that need exact
  Spotify metadata, treat Last.fm as the user-side history and the Spotify
  Tracks Dataset as a separate item universe.

## Dataset 3: Sguerra Pre-Generated NL Profiles (already in repo)

The `code/recsys25_llm_biases/data/` directory (cloned, not downloaded) ships
with two CSVs from the RecSys 2025 paper:
- `user_data.csv`: 1488 rows of (user × LLM × time-window) NL profiles, user
  ratings, and downstream Recall@10/NDCG@10. 64 real Deezer users.
- `long_term.csv`: 64 long-term genre/country distributions per user.

These are ready-to-use for replicating the bias analysis without running any
LLM API calls.

## Recommended split for this study

For a comparison of "LLM given Spotify export" vs "Spotify-style algorithm",
the experiment runner should:
1. Pick N=20-50 Last.fm users.
2. For each, hold out the last 30 days of listens as ground truth.
3. Build candidate pool: hold-outs ∪ random/popularity-sampled negatives from
   the Spotify Tracks Dataset.
4. Run two recommenders over the same pool:
   - **LLM (Claude)**: prompt with listening history → ranked recommendations.
   - **Baseline**: content-based cosine on audio features (Boadana et al.
     2025) and/or item-kNN/EASE collaborative filter (Sanner et al. 2023).
5. Report Recall@10, NDCG@10, novelty, popularity bias (ARP), and genre
   diversity — the dimensions Epure et al. 2025 identify as critical for LLM
   music recommenders.
