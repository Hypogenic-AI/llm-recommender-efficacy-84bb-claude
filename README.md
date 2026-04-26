# LLM-vs-Spotify Recommender Efficacy

A head-to-head study testing whether **Claude Sonnet 4.5**, given a user's
exported listening history, ranks music recommendations as well as the
classical algorithms that historically powered Spotify's "Discover Weekly"
(item-kNN collaborative filtering, content-based filtering, popularity).

## Key findings (N=24 users, Last.fm 1K → Spotify catalog)

- **Claude with the user's history is the joint-best ranker** by NDCG@10
  (0.375) and tied with GPT-4.1 (0.376), beating CBF (0.333), item-kNN
  (0.306), and popularity (0.168, p=0.028, d=0.53).
- **Personalization gain is large and significant**: NDCG@10 +0.29,
  p=0.002, Cohen's d = 0.86 between Claude-with-history and Claude-no-history
  — the LLM is genuinely conditioning on the listening log, not relying on
  popularity priors.
- **No popularity shortcut**: Claude's ARP@10 (58.6) is *below* the
  popularity baseline (74.3), in line with item-kNN.
- **Cross-LLM robustness**: GPT-4.1 with the same prompt achieves
  statistically indistinguishable accuracy at ≈3× lower latency.
- **Hallucination on free-generation**: 50 % of Claude's open-form
  recommendations resolve to the 77K-track catalog (38 % for GPT-4.1) —
  partly an artifact of catalog size, but flags real catalog grounding as a
  limitation.
- **Cost**: $0.45 total for the entire study (168 LLM calls).

See [`REPORT.md`](REPORT.md) for the full writeup.

## Direct answer to the user's question

> *If I export all my Spotify data and ask Claude for recommendations, is it
> better than Spotify recommendations?*

**On objective held-out-track accuracy, Claude (with your history) matches
or beats every classical Spotify-style algorithm we could test offline.**
The improvement over the strongest classical competitor (TF-IDF
content-based filtering) is small (+0.04 NDCG@10) and not significant on
N=24, but the LLM never loses. Whether it beats Spotify's *current
production* recommender — a closed black box that blends CF, CBF, audio
embeddings, and contextual bandits — is not something an offline study can
answer.

## Reproduction

Requires `OPENROUTER_KEY` and `OPENAI_API_KEY` in environment.

```bash
uv venv && source .venv/bin/activate
uv add pandas datasets scikit-learn scipy matplotlib openai pyarrow

# Datasets are downloaded already in datasets/; if not, see datasets/README.md.

python src/run_experiment.py --n_llm_runs 3 --test_days 30   # ~25 min, $0.40
python src/run_hallucination.py                              # ~3 min, $0.05
python src/analyze.py                                        # 5 sec
python src/extra_analysis.py                                 # 1 sec
```

## Files

```
planning.md              Research plan with motivation, hypotheses, methods
literature_review.md     Synthesis of the LLM-recommender literature (gathered)
resources.md             Catalog of datasets, papers, and code (gathered)
REPORT.md                Final research report with results
src/                     Implementation
  data_loader.py         Last.fm + Spotify catalog loaders + train/test splits
  build_pools.py         Per-user 20-track candidate pool builder
  baselines.py           random / popularity / item-kNN / CBF rankers
  llm_ranker.py          Claude / GPT-4.1 zero-shot ranker (Hou et al. 2023)
  metrics.py             Recall@k, NDCG@k, MRR, ARP@k, LongTailShare,
                         GenreDiversity, EntitiesResolvedShare
  run_experiment.py      Main orchestrator
  run_hallucination.py   Free-generation hallucination diagnostic
  analyze.py             Wilcoxon, bootstrap CIs, plots
  extra_analysis.py      Personalization gain, run variance, cost
datasets/                Spotify Tracks Dataset + Last.fm 1K (50-user sample)
papers/                  19 PDFs from the literature review
code/                    6 cloned reference repos (LLMRank, recsys25_llm_biases, …)
results/                 CSV outputs and JSON dumps
figures/                 Bar/box plots used in REPORT.md
logs/                    Run logs for reproducibility
```
