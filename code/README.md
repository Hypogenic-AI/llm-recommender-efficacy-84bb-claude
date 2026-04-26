# Cloned Repositories

Six repositories were cloned to support the experiment runner. The first one
(deezer/recsys25_llm_biases) is the most directly applicable: it ships ready-to-use
NL profiles and an end-to-end downstream evaluation pipeline.

## 1. recsys25_llm_biases (Deezer / Sguerra et al. 2025)

- **URL**: https://github.com/deezer/recsys25_llm_biases
- **Paper**: Biases in LLM-Generated Musical Taste Profiles for Recommendation
  (RecSys 2025) — arxiv 2507.16708
- **Why it matters**: Most directly aligned with our hypothesis. Generates
  natural-language taste profiles from listening history and evaluates them
  both via user ratings and a downstream recommendation task.
- **Ready-made data**: `data/user_data.csv` (1488 profile×rating rows from 64
  real Deezer users, three LLMs, four time windows) and `data/long_term.csv`.
- **Key entry points**:
  - `src/doubly_robust.py` — Doubly Robust ATE estimation for genre/country bias.
  - `src/eval.py` — downstream Recall@10 / NDCG@10 evaluation.
  - `src/evaluator.py` — bi-encoder scoring against held-out items.
  - `LLM_bias_plots.ipynb` — paper figures.
- **Application**: Use the profile-generation pattern (top-N artists per
  user × window → LLM prompt → NL profile → bi-encoder retrieval) directly.
  The pre-generated profiles also let us start with downstream-only experiments
  before paying for API calls.
- **Install**: `make build` then `make run-bash` (Docker). For host-side use,
  install `pandas`, `numpy`, `scikit-learn`, `sentence-transformers`,
  `torch`, `econml` (per the Dockerfile).

## 2. LLMRank (Hou et al. 2023)

- **URL**: https://github.com/RUCAIBox/LLMRank
- **Paper**: Large Language Models are Zero-Shot Rankers for Recommender
  Systems (ECIR 2024) — arxiv 2305.08845, 519 citations.
- **Why it matters**: Foundational framework for using LLMs as zero-shot
  rankers given a candidate pool. Establishes prompting patterns for
  sequential history and candidate ranking, plus diagnostics for popularity
  bias and position bias.
- **Key entry points**:
  - `llmrank/` — main package with prompt templates and ranking pipeline.
  - `scripts/` — bash entry points to reproduce experiments on MovieLens/Games.
  - `requirements.txt` — pinned dependencies (RecBole, OpenAI client, etc.).
- **Application**: Adopt their three prompting variants (sequential,
  recency-focused, in-context-learning) and bootstrapping for position bias
  when the LLM ranks candidate tracks for our users.

## 3. LLM4RS (Dai et al. 2023)

- **URL**: https://github.com/rainym00d/LLM4RS
- **Paper**: Uncovering ChatGPT's Capabilities in Recommender Systems
  (RecSys 2023) — arxiv 2305.02182, 364 citations.
- **Why it matters**: Tests point-wise, pair-wise, and list-wise prompting on
  five domains (movies, books, music, news, products). Provides a clean
  reference for how to structure each prompting style and how to compare LLM
  output against traditional rankers.
- **Application**: Their three ranking modes give us a template for the
  evaluation matrix; list-wise had the best cost/quality trade-off.

## 4. LLM-Next-Item-Rec (Wang & Lim 2023)

- **URL**: https://github.com/AGI-Edgerunners/LLM-Next-Item-Rec
- **Paper**: Zero-Shot Next-Item Recommendation using Large Pretrained
  Language Models — arxiv 2304.03153.
- **Why it matters**: Defines the NIR (Next-Item Recommendation) prompting
  strategy with an external candidate-generation module — exactly the
  retrieve-then-rank pattern we plan for our LLM vs Spotify-algorithm
  comparison.
- **Application**: Use their NIR prompt template as the "no-extras" baseline
  prompt for Claude.

## 5. LLMs-as-Zero-Shot-Conversational-RecSys (He et al. 2023)

- **URL**: https://github.com/AaronHeee/LLMs-as-Zero-Shot-Conversational-RecSys
- **Paper**: Large Language Models as Zero-Shot Conversational Recommenders
  (CIKM 2023) — arxiv 2308.10053, 247 citations.
- **Why it matters**: Largest public conversational recommendation dataset
  scraped from Reddit, plus probing tasks that diagnose what LLMs actually use
  (titles vs descriptions vs popularity priors). Useful even if our task is
  not conversational — the probing methodology transfers.
- **Application**: Their popularity-prior probing test is a good diagnostic
  to add to our evaluation alongside Recall/NDCG.

## 6. Text2Tracks (Spotify / Palumbo et al. 2025)

- **URL**: https://github.com/mayurbhangale/text2tracks
- **Paper**: Text2Tracks: Prompt-based Music Recommendation via Generative
  Retrieval — arxiv 2503.24193.
- **Why it matters**: Direct industry comparison from Spotify Research. Trains
  a generative retrieval model to map natural-language prompts to track IDs;
  shows semantic IDs beat title-string IDs by 127%.
- **Application**: Reference for the right way to encode track identifiers if
  we extend beyond off-the-shelf prompting. Also clarifies why pure
  natural-language prompting (titles in, titles out) underperforms.

## Setup notes

All repos are pinned to `--depth 1`. Re-clone with full history if blame /
log is needed. None of the repos modifies the workspace `pyproject.toml`;
each can be installed in its own sub-environment if needed.

The recsys25_llm_biases repo is the only one we have actually exercised so
far (verified `data/user_data.csv` parses and contains 1488 rows of
pre-generated profiles + ratings). LLMRank, LLM4RS, NIR, and LLMs-Conv-RecSys
have READMEs that match the published methodology and standard install
instructions; the experiment runner should test-install each in turn before
running.
