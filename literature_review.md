# Literature Review: How much do LLMs solve recommendation algorithms?

**Hypothesis under test**: When given exported Spotify data, an LLM (e.g.,
Claude) can produce music recommendations as good as or better than Spotify's
own recommender.

## Research area overview

The use of LLMs as recommenders has gone from "promising preliminary results"
in early 2023 (Hou et al., Dai et al., Sanner et al., Wang & Lim) to a mature
sub-field with surveys (Lin et al. 2023; Epure et al. 2025), benchmarks
(RecBench/2503.05493), and music-specific systems (TalkPlay family, Text2Tracks
from Spotify Research, Sguerra et al., Boadana et al.). The literature has
converged on three settings:

1. **LLM as zero-shot ranker over a candidate pool** — items are pre-retrieved
   by a traditional method; the LLM only re-ranks (Hou et al. 2023, Dai et al.
   2023). This is currently the most cost-effective and reproducible setup.
2. **LLM as a natural-language profile generator** — the LLM summarizes a
   user's history into a scrutable text profile that is then consumed by a
   downstream retrieval model (Radlinski et al. 2022; Sguerra et al. 2025;
   Sanner et al. 2023).
3. **LLM as the end-to-end recommender** — the LLM ingests history + catalog
   and emits track-level recommendations directly, often as conversational
   agents (TalkPlay; Yun et al. 2025; Boadana et al. 2025).

For our hypothesis, setting 3 is the most direct test, while setting 1 is the
fairest comparison against Spotify's algorithm because it isolates the
ranking decision from candidate generation.

## Key papers (deep reads)

### Sguerra, Epure, Lee, Moussallam (2025) — Biases in LLM-Generated Musical Taste Profiles
*RecSys 2025; arXiv 2507.16708; Deezer Research.*

- **Setup**: 64 Deezer users; for each, sample top-N artist-track pairs from
  four time windows (30/90/180/365 days), prompt three LLMs (Llama 3.2,
  DeepSeek-R1, Gemini 2.0 Flash) to produce an NL taste profile, ask the user
  to rate their own profile (with random profiles as negatives), and embed the
  profile + held-out tracks into a shared latent space (bi-encoder fine-tuned
  with MS-MARCO cross-encoder as teacher) for downstream Recall@10 / NDCG@10.
- **Findings**:
  - Self-identification scores are higher than random baselines for all
    three models, but choice of LLM dominates over time-window length.
  - Specialist users (high GS-score) get more representative profiles than
    generalists.
  - Item-side biases are real and persist after Doubly Robust ATE: rap content
    *lowers* ratings; metal and US-origin content *raises* them.
  - Self-identification correlates only weakly with downstream Recall@10 — so
    "users like the profile" ≠ "the profile makes good recommendations".
- **Code/data available**: `github.com/deezer/recsys25_llm_biases` ships
  pre-generated profiles + ratings + downstream metrics for all 1488
  user×model×window combinations.
- **Why this matters for us**: This is the closest published precedent. The
  prompt template, sampling strategy (top-N artists × top tracks per artist),
  and downstream evaluation pipeline are reusable. The decoupling of
  user-perceived quality and downstream accuracy is a critical caveat for our
  evaluation design.

### Boadana et al. (2025) — LLM-Based Intelligent Agents for Music Recommendation
*arXiv 2508.11671; in Portuguese.*

- **Setup**: Real Spotify data collected over 13 months from 19 voluntary
  users (22,178 unique tracks). 300-track sampled catalog. Compares CrewAI
  multi-agent system on Gemini 2.0 Flash and LLaMA-3.3-70B (via Groq) against
  a TF-IDF / cosine content-based baseline. Blind 10-user evaluation; each
  user rates 3 playlists × 10 tracks on (liked, known, overall rating 0-10).
- **Findings**:

  | Model | Like Rate | Novelty | Successful Novelty | Rating | Latency |
  |---|---|---|---|---|---|
  | CBF (TF-IDF) | 61% | 58.5% | 21.0% | 6.70 | 1.4 s |
  | LLaMA 3.3 70B | **89.32%** | 11.85% | 3.17% | **8.70** | 84 s |
  | Gemini 2.0 Flash | 65% | 52.0% | 18.5% | 7.25 | 71 s |

- **Why this matters for us**: This is the closest existing test of our
  hypothesis. Headline answer: yes, LLMs can produce *more-liked* playlists
  from Spotify export data, but they trade off discovery (lower novelty) and
  pay a 50–60× latency cost. The comparison is against CBF, not Spotify's
  actual algorithm — but Spotify's production system blends CBF, CF, and
  context, so this is a reasonable lower-bound competitor.

### Sanner, Balog, Radlinski, Wedin, Dixon (2023) — LLMs are Competitive Near Cold-start Recommenders
*RecSys 2023; arXiv 2307.14225; Google.*

- **Setup**: New parallel corpus on movies. Two-phase user study: (1) elicit
  five liked + five disliked items *and* free-text +/− preference paragraphs;
  (2) blind-rate 40 candidate items drawn from 4 pools — 10 popular random,
  10 mid-popular random, 10 from EASE collaborative filter, 10 from BM25
  Late-Fusion text retrieval over Amazon Movie Reviews.
- **Findings**: For *language-only* preferences (no item history), LLM
  prompting matches item-based CF in the near cold-start regime — even though
  the LLM has zero supervised training for the task. Adding item examples
  helps further. Dispreferences modestly improve language-based recommendation.
- **Why this matters for us**: The two-phase elicit-then-blind-rate study is
  the gold-standard design we should adopt. The unbiased random pool ensures
  comparison across methods is unbiased by any single retriever.

### Hou et al. (2023) — LLMs are Zero-Shot Rankers for Recommender Systems
*ECIR 2024; arXiv 2305.08845; Renmin U + UCSD; 519 citations.*

- **Setup**: Formalize recommendation as conditional ranking. LLM is fed
  sequential interaction history (the "condition") + a candidate pool (20
  items, mixed retriever sources), asked to rank. Three prompt variants:
  sequential, recency-focused, in-context-learning.
- **Findings**: LLMs (GPT-3.5) have non-trivial zero-shot ranking ability and
  *can challenge conventional models when candidates are diverse*, but suffer
  from (a) inability to perceive history order without explicit prompts, (b)
  position bias in candidate ordering (alleviated by bootstrapping multiple
  permutations), (c) popularity bias (alleviated by the right prompts).
- **Why this matters for us**: Provides the prompting templates we should
  adopt verbatim, plus the diagnostic protocol for popularity / position bias
  that we *must* run for our LLM recommender to be a credible competitor.

### Epure, Deldjoo, Sguerra, Schedl, Moussallam (2025) — Music Recommendation with LLMs
*arXiv 2511.16478; 36-page survey; Deezer + JKU + PoliBari.*

- **Setup**: Position paper / survey. Reviews how LLMs reshape user modeling,
  item modeling, and natural-language recommendation in music. Synthesizes an
  evaluation framework drawing from NLP.
- **Evaluation framework adopted by this study** (sections 3.2.1–3.2.2):
  - **G1. Query Adherence & Groundedness** — KB entity resolution,
    constraint satisfaction, hallucination penalty.
  - **G2. Discovery Quality** — LongTailShare, Diversity@k, GeoDiv@k.
  - **G3. Personalization Gain** — uplift with vs. without user context.
  - **G4. Profile Fidelity & Controllability** — alignment, edit responsiveness.
  - **G5. Cultural / Linguistic Coverage** — region/language representation.
  - **G6. Classical Relevance** — Precision@k, Recall@k, nDCG (read alongside G1–G5).
  - Risk diagnostics: hallucinations, popularity/temporal/language bias
    (ARP from Abdollahpouri et al.), profile hazards (RBO under perturbations),
    evaluator bias when LLM-as-judge is used.
- **Why this matters for us**: This is the richest single source for *what to
  measure*. It directly addresses the methodological question of how to
  evaluate generative recommenders fairly. We should adopt at least G1, G2,
  G3, G6 and the popularity-bias and hallucination diagnostics.

## Common methodologies across the field

| Methodology | Used in |
|---|---|
| Zero-shot prompting with item titles + history | Hou et al. 2023; Dai et al. 2023; Wang & Lim 2023; Boadana et al. 2025 |
| Few-shot in-context learning | Hou et al. 2023; Sanner et al. 2023 |
| LLM-generated NL user profile → bi-encoder retrieval | Sguerra et al. 2025; Sanner et al. 2023; Radlinski et al. 2022 |
| Multi-agent LLM with tool calling | Boadana et al. 2025; TalkPlay-Tools (Doh et al. 2025) |
| Generative retrieval with semantic IDs | Text2Tracks (Palumbo et al. 2025) |
| Chain-of-thought / reasoning over preferences | Recent (Bringing Reasoning to Generative Rec, 2026) |

## Standard baselines

| Baseline | Used in |
|---|---|
| Item-kNN | Hou et al. 2023; LLMRank |
| EASE (closed-form autoencoder CF) | Sanner et al. 2023; Steck 2019 |
| BERT4Rec / SASRec sequential models | Hou et al. 2023; RecBench |
| TF-IDF + cosine content-based | Boadana et al. 2025 |
| BM25 over item descriptions | Sanner et al. 2023; Penha & Hauff 2020 |
| Random / popularity (sanity baselines) | All studies |

For our Spotify comparison, the production-grade equivalent of Spotify's
"Discover Weekly" / "Made For You" cannot be re-implemented offline.
Reasonable proxies are: (a) item-kNN on track co-occurrence, (b) CBF on audio
features (Boadana et al. proxy), (c) popularity-stratified random.

## Evaluation metrics

| Metric | Captures | Source |
|---|---|---|
| Recall@10, NDCG@10 | Classical relevance | Hou et al.; Sguerra et al.; RecBench |
| Hit Rate / Like Rate | Subjective satisfaction | Boadana et al. |
| Novelty Rate | Fraction of items new to user | Boadana et al. |
| Successful Novelty Rate | Liked × novel | Boadana et al. |
| ARP (Avg Rec Popularity) | Popularity bias | Abdollahpouri et al. via Epure et al. |
| Diversity@k, GeoDiv@k | Discovery quality | Epure et al. |
| EntitiesResolvedShare | Hallucination rate | Epure et al. |
| Rank-Biased Overlap | Stability under perturbation | Epure et al. |

## Datasets in the literature

| Dataset | Used in |
|---|---|
| Spotify Million Playlist Dataset (MPD) | RecSys '18 challenge; many follow-ons (no longer public) |
| Last.fm 1K Users / LFM-1b / LFM-2b | Schedl et al.; many MIR/RecSys studies |
| 30Music | Turrin et al. 2015 |
| Million Song Dataset + Last.fm tags | Bertin-Mahieux et al. 2011 |
| MovieLens | Hou et al.; Sanner et al.; Kang et al. (cross-domain proxy) |
| Deezer internal logs | Sguerra et al.; Tran et al. |
| Custom Spotify exports | Boadana et al. (19 users) |

## Gaps and opportunities

1. **No published study compares an LLM directly against the Spotify
   production algorithm.** Boadana et al. is the closest but uses a TF-IDF
   CBF baseline. This is the gap our research targets.
2. **Self-identification ≠ downstream accuracy.** Sguerra et al. show that
   user-perceived profile quality and Recall@10 are only weakly correlated.
   Any user study should measure both.
3. **Popularity/exposure bias is consistently reported in LLM
   recommenders** (Hou et al.; Sguerra et al.; Deldjoo 2024). Our LLM arm
   needs an ARP / long-tail-share diagnostic to be a credible competitor.
4. **Cultural and linguistic biases** — Sguerra et al. find rap content
   reduces profile quality. Any large-scale evaluation should stratify
   results by genre.
5. **Hallucination** is rarely measured (most studies ignore the rate at
   which LLM-output tracks don't exist in the catalog). Our evaluation must
   match outputs to the catalog and report EntitiesResolvedShare.
6. **Cost / latency**: Boadana et al. report 50–84× latency penalty.
   Worth tracking inference cost and time as a "soft" constraint metric.
7. **Reproducibility under model drift**: Epure et al. note LLMs lack
   train/test separation guarantees because public datasets may be in
   pre-training. Use post-cutoff data where possible, or new user-collected
   exports.

## Recommendations for our experiment

Based on the literature, the experiment runner should design the study as
follows:

### Recommended datasets
- **Primary**: Last.fm 1K-user dataset (50-user sample is sufficient for an
  initial study; full 1K for statistical power). Listening events are
  per-user time-stamped, mirroring Spotify export semantics.
- **Item catalog**: HuggingFace `maharshipandya/spotify-tracks-dataset` (114K
  Spotify tracks with genre + audio features) for candidate retrieval and
  popularity stratification.
- **Optional**: The pre-generated NL profiles in
  `code/recsys25_llm_biases/data/` let us test the downstream evaluation
  pipeline before paying for any LLM API calls.

### Recommended baselines
- **EASE** (closed-form autoencoder CF) — strong, simple, used by Sanner
  et al. as their item-CF reference.
- **Item-kNN** on track co-occurrence — fastest baseline, widely reported.
- **TF-IDF / cosine over genre tags** — Boadana et al. CBF baseline; closest
  hand-crafted analogue to "what Spotify does with metadata".
- **Popularity-only baseline** — sanity check.

### Recommended LLM arms
- **Zero-shot ranking with sequential prompt** (Hou et al. template) — give
  Claude the listening history + a 20-track candidate pool, ask for a ranking.
- **NL-profile-then-retrieval** (Sguerra et al. template) — Claude generates a
  taste profile, bi-encoder retrieves tracks. Decouples reasoning from
  retrieval.
- Optional: **Tool-use agent** — Claude calls Spotify-style metadata search
  tools to ground recommendations (TalkPlay-Tools-style).

### Recommended metrics (mandatory)
- Recall@10, NDCG@10 (G6).
- ARP (popularity bias), LongTailShare (G2 + risk).
- EntitiesResolvedShare (hallucination diagnostic).
- Genre Diversity@10 (G2).
- Personalization Gain (Recall with profile vs without — quantifies what the
  LLM actually learned from the user data).

### Recommended metrics (optional, if user study is feasible)
- Like Rate / Playlist Rating (Boadana et al. style) — captures the subjective
  aspect that downstream metrics miss.
- Successful Novelty Rate — joint discovery + satisfaction.

### Methodological considerations
- **Position bias**: Bootstrap candidate orderings for the LLM ranker (Hou et
  al.).
- **Catalog grounding**: Always parse LLM output against the candidate set;
  count tracks the LLM mentioned but that don't exist in our catalog as
  hallucinations.
- **Train/test contamination**: The LLM may have seen Last.fm during
  pre-training. Hold out the most recent ~30 days of listening per user as
  the test set; do *not* feed those into the prompt.
- **Cost transparency**: Log token counts and latency per recommendation so
  the trade-off can be reported alongside accuracy.
- **Stratify reporting** by genre and popularity tier — Sguerra et al. show
  aggregate metrics hide systematic disparities.
- **Document model + temperature**: LLM arms are non-deterministic; report
  median across ≥3 runs at temperature 0.0 for the primary results, and
  show variance at temperature 0.7 as a robustness check.
