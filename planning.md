# Research Plan: How much do LLMs solve recommendation algorithms?

## Motivation & Novelty Assessment

### Why This Research Matters
The user-submitted question — "If I export all my Spotify data and ask Claude
for recommendations, is it better than Spotify recommendations?" — is the
end-user's version of an active open question in the recommender systems
community. If the answer is "yes, even the off-the-shelf foundation model is
competitive," then there are immediate consumer-facing implications (any user
with their data could opt out of platform-locked recommendations) and research
implications (much of the engineering investment in collaborative filtering may
be replaceable by prompting). If the answer is "no, classical CF still wins,"
that result is also valuable — it pushes back against current LLM enthusiasm in
the recommender field with a clean, well-controlled comparison.

### Gap in Existing Work
Per `literature_review.md`:
1. **No published study compares an LLM directly against the Spotify production
   recommender.** Boadana et al. (2025) — the closest precedent — uses a
   TF-IDF content-based filter (CBF) baseline, not Spotify's actual algorithm.
   Spotify's production recommender is a closed black box that we cannot run
   offline; the literature consistently uses CF baselines (item-kNN, EASE) and
   CBF baselines as "Spotify-like" proxies.
2. **Self-identification ≠ downstream accuracy.** Sguerra et al. (2025) show
   user-perceived profile quality and recall@10 are only weakly correlated.
3. **Popularity bias is consistently reported but rarely diagnosed**
   alongside accuracy.
4. **Hallucination is rarely measured.**

### Our Novel Contribution
We run a head-to-head comparison on real listening histories with:
1. **The exact prompting protocol from Hou et al. (2023)** for zero-shot LLM
   ranking, applied to **Claude Sonnet 4.5** — the user's named model — and
   **GPT-4.1** as a cross-model robustness check.
2. **Three classical baselines that operationally proxy Spotify's algorithmic
   stack**: popularity-only, item-kNN (collaborative filtering), and TF-IDF
   over genre tags + audio features (content-based).
3. **The full Epure et al. (2025) evaluation framework on a single benchmark**:
   classical relevance (recall@k, NDCG@k) + popularity bias (ARP,
   long-tail-share) + hallucination (entities-resolved-share) + diversity
   (genre diversity@10).
4. **A reproducible, end-to-end Last.fm-as-Spotify-export pipeline** that
   anyone can re-run with API credits.

The contribution is not a new algorithm — it is a clean, multi-metric,
multi-model evaluation in the specific operational regime the user asked about
(give the LLM your full listening history, ask for recommendations, compare to
the platform-provided recommendations).

### Experiment Justification
- **Experiment 1 — End-to-end ranking comparison**: The core test of the
  user's question. For each user, we hold out the last week of listens, build
  a 20-track candidate pool (with positives mixed in among popular and
  CF-retrieved items), and score each method's ability to rank the held-out
  positives high. Why needed: this is what the hypothesis literally asks.
- **Experiment 2 — Popularity-bias diagnostic (ARP, long-tail-share)**: LLM
  recommenders are documented to over-recommend popular items (Hou et al.
  2023; Sguerra et al. 2025). Why needed: a method can win on recall by
  recommending only popular items — we need to verify it isn't doing that.
- **Experiment 3 — Hallucination rate (entities-resolved-share)**: When the
  LLM is allowed to free-generate (rather than rank a closed candidate pool),
  some output tracks may not exist in the catalog. Why needed: the Epure
  et al. (2025) framework requires this; without it the ranking-only setup
  hides a real failure mode.
- **Experiment 4 — Personalization-gain ablation**: Run Claude with vs.
  without the user's history. Why needed: this isolates how much of the LLM's
  performance is real personalization vs. priors over popular tracks.
- **Experiment 5 — Cost / latency / variance accounting**: Token counts,
  wall-clock latency, and inter-run variance at temperature 0.0 (median of 3
  runs). Why needed: any "LLM beats CF" claim must be paired with the
  practical cost of that win.

## Research Question

**Does a state-of-the-art LLM (Claude Sonnet 4.5), given a user's listening
history in the format of a Spotify export, produce better music
recommendations than classical recommender baselines that approximate the
Spotify production stack?**

We restrict "better" to four operationalizations: (a) higher Recall@10 and
NDCG@10 against a held-out portion of each user's actual subsequent listens;
(b) lower popularity bias (lower ARP, higher long-tail-share); (c) higher
genre diversity at the same accuracy; (d) low hallucination rate. We also
report cost and latency.

## Hypothesis Decomposition
- **H1 (relevance)**: Claude's ranked recommendations achieve Recall@10 and
  NDCG@10 at least as high as the best classical baseline.
- **H2 (no popularity shortcut)**: Claude's ARP is not above the popularity
  baseline; long-tail-share is non-negligible.
- **H3 (low hallucination)**: When Claude is asked to free-recommend rather
  than rank, ≥80% of its tracks resolve to the catalog.
- **H4 (real personalization)**: Recall improves when Claude is given the
  user's history vs. when it is given no history (personalization gain > 0).

Independent variable: recommender method (popularity, item-kNN, TF-IDF CBF,
Claude zero-shot ranker, GPT-4.1 zero-shot ranker, Claude no-history
ablation).
Dependent variables: Recall@10, NDCG@10, ARP, LongTailShare,
GenreDiversity@10, EntitiesResolvedShare, latency, cost.

## Methodology

### Approach
Adopt **Setting 1 from the literature review (LLM as zero-shot ranker over a
candidate pool)** as the primary, fair comparison. Reasons:
1. Apples-to-apples comparison: every method ranks the same candidate pool, so
   differences are attributable to the ranking decision, not to candidate
   generation.
2. Bounds catalog size and prevents the LLM from hallucinating tracks that
   don't exist (which is a separate failure mode we measure in Experiment 3).
3. Cost-effective: 20-item pools per user × ~50 users × 6 methods is small
   enough to run on API credits in this session.

For Experiment 3 we additionally run Claude in the **free-generation
("setting 3") mode** to measure the hallucination rate the ranking-only setup
papers over.

### Datasets
- **Listening history (proxy for Spotify export)**: `datasets/lastfm_1k/` —
  50-user time-stamped Last.fm listening logs. Each row is `(user_id, ts,
  artist, track)`.
  - Per user, sort by timestamp ascending.
  - Train window: all listens except the last 7 days.
  - Test window: the last 7 days. Positives = unique tracks listened to in
    the last 7 days that the user did not listen to in the train window
    (definition of "discovery" relative to history).
- **Item catalog**: `datasets/spotify_tracks_dataset/` (114K tracks, with
  audio features and genre). We project Last.fm tracks onto Spotify tracks by
  fuzzy-joining on `(artist, track)` lowercased; tracks that don't match are
  dropped from the candidate pool but kept in user history (used as text
  context for the LLM).

### Candidate Pool Construction
For each (user, test-window) we build a 20-track candidate pool:
- 1–N positives (held-out tracks for that user that resolved against the
  catalog), capped at 5 to avoid trivializing the ranking task.
- Item-kNN candidates: top-K co-occurring tracks with the user's last 50
  history tracks. Drop those already in user history. Add up to 8 such
  candidates.
- Popularity candidates: top global tracks (filtered by genre overlap with
  the user's top-3 history genres) not already in history. Fill the
  remaining slots up to 20.
- All candidates are deduplicated and shuffled with a per-user random seed
  (so position is randomized across methods).

### Methods
1. **Random** — sanity baseline, ranks the 20 candidates uniformly at
   random.
2. **Popularity** — ranks candidates by their `popularity` score from the
   Spotify Tracks Dataset (descending).
3. **Item-kNN** — for each candidate, score = sum of co-occurrence with
   the user's last 50 history items in the train window. Co-occurrence is
   computed via item-item co-listening on Last.fm history at user-level.
4. **CBF (TF-IDF over genre + audio bins)** — Boadana et al. analogue. Each
   track is represented by its genre tag + binned (0.2-step) audio features
   (danceability, energy, valence, acousticness). Score = cosine similarity
   to the user's mean track vector over the last 50 history tracks.
5. **Claude (Sonnet 4.5) zero-shot ranker** — Hou et al. sequential prompt:
   user history (last 50 tracks, formatted as "1. Artist - Track") + 20
   shuffled candidates, "rank the candidates in order of how likely the user
   is to enjoy them" with 1-line per ranked candidate, parsed back to indices.
   Temperature 0.0. Three runs (median).
6. **GPT-4.1 zero-shot ranker** — same prompt as #5, different model.
7. **Claude no-history (ablation)** — same prompt as #5 but with only the
   candidates and instruction "rank these tracks by general popularity and
   musical merit". Establishes the no-personalization floor for the LLM.

### Evaluation Metrics
- **Recall@10**: |{retrieved positives in top 10}| / |held-out positives|.
- **NDCG@10**: standard DCG with binary relevance, normalized by IDCG.
- **MRR**: mean reciprocal rank of the first true positive.
- **ARP (Average Recommendation Popularity)**: mean Spotify popularity
  score of items ranked in top 10 (Abdollahpouri et al.).
- **LongTailShare@10**: fraction of top-10 items whose Spotify popularity
  is below the global 50th percentile.
- **GenreDiversity@10**: number of distinct `track_genre` values among the
  top 10 / 10.
- **EntitiesResolvedShare** (Experiment 3 only, free-generation mode):
  fraction of Claude's generated track names that resolve to a track in the
  catalog via fuzzy match.
- **Latency / Cost**: wall-clock seconds per user; total $ spent (estimated
  from token counts × OpenRouter pricing).

### Statistical Analysis
- For each metric we have one value per user. We compare each LLM arm vs.
  each baseline using a **paired Wilcoxon signed-rank test** (non-parametric,
  no normality assumption, fits per-user paired structure).
- We report **median, mean, IQR, and 95% bootstrap CI** for each metric.
- We do **NOT** Bonferroni-correct across the 4 hypotheses because they are
  pre-registered as separate hypotheses (each tested at α=0.05) and address
  different questions; we will note this transparently.

### Sample Size & Power
50 users × 6 methods × 3 LLM runs (for the LLM arms only; classical
baselines are deterministic) is small but adequate for a within-user paired
comparison. With Wilcoxon and N=50, we have ~80% power to detect
small-to-medium per-user effect sizes (d≈0.4) at α=0.05.

## Expected Outcomes
Based on Boadana et al. (2025) and Hou et al. (2023):
- **Likely**: Claude wins on subjective like-rate proxies; classical CF wins
  or ties on Recall@10 if the candidate pool is heavily co-occurrence-driven.
- **Likely**: Claude shows higher ARP (popularity bias) than item-kNN.
- **Likely**: Claude has high entities-resolved-share (>80%) on common
  artists, lower on long-tail.
- **Robustness**: Personalization gain should be > 0 (Claude with history
  > Claude without).

## Timeline (rough)
- Phase 0 (motivation + plan): 30 min — done now.
- Phase 2 (env, data load, pool build): 30 min.
- Phase 3 (implement baselines + LLM arms): 60 min.
- Phase 4 (run experiments — main bottleneck is LLM API time): 45 min.
- Phase 5 (analysis): 30 min.
- Phase 6 (writeup): 30 min.
- Total: ~3.5 hours.

## Potential Challenges
- **Catalog matching loss**: Last.fm artist/track strings may not match the
  Spotify catalog exactly. Mitigation: lowercase + fuzzy match; report match
  rate; require ≥1 positive per user (drop users with 0 matches).
- **API rate limits / cost**: Per-user prompt is ~1K input tokens, 50 users
  × 3 runs × 2 LLMs ≈ 300 API calls; well under any reasonable rate limit.
  Cost estimate: ~$3–5 total.
- **LLM output parsing failures**: The model may not produce a clean ranked
  list. Mitigation: Hou et al. parsing fallback (regex-extract numbered list,
  fall back to lexical match against candidate strings, ultimate fall-back
  to popularity ranking with logging).
- **Position bias**: Mitigated by random shuffling per user × per run.

## Success Criteria
The research succeeds (i.e., produces reliable knowledge) regardless of which
direction the result points, as long as:
1. All four metrics (relevance, popularity, hallucination, personalization
   gain) are reported with effect sizes and statistical tests.
2. We compare against ≥3 classical baselines.
3. We test ≥2 LLMs to show the result isn't an artifact of one model.
4. We document and report cost, latency, and reproducibility info.
