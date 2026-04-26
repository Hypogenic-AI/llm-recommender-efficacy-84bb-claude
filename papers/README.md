# Downloaded Papers

19 papers, organized into three groups: foundational LLM-as-recommender work
(2023, high citation), music-specific LLM recommendation, and supporting
context. All are arXiv preprints (free, no paywall).

## Foundational LLM-as-recommender (general)

1. **2305.08845** Hou et al. — Large Language Models are Zero-Shot Rankers for
   Recommender Systems (ECIR 2024, 519 citations). Defines the prompting
   templates (sequential, recency-focused, ICL), surfaces popularity and
   position bias, shows bootstrapping fixes. Code: github.com/RUCAIBox/LLMRank.
2. **2305.02182** Dai et al. — Uncovering ChatGPT's Capabilities in Recommender
   Systems (RecSys 2023, 364 citations). Compares point-wise, pair-wise,
   list-wise prompting across five domains including music. Code: rainym00d/LLM4RS.
3. **2305.06474** Kang et al. — Do LLMs Understand User Preferences? Evaluating
   LLMs On User Rating Prediction (2023, 172 citations). Zero-shot LLMs
   underperform CF on rating prediction; fine-tuning closes the gap with little
   data. Important counterweight to optimistic results.
4. **2308.10053** He et al. — Large Language Models as Zero-Shot Conversational
   Recommenders (CIKM 2023, 247 citations). Largest conversational rec dataset;
   probing tasks for what the LLM actually uses.
5. **2307.14225** Sanner et al. — LLMs are Competitive Near Cold-Start
   Recommenders for Language- and Item-based Preferences (RecSys 2023, 185
   citations). User study: LLMs match item-based CF when given language
   descriptions instead of items. The two-phase study design (elicit →
   blind-rate) is the closest template for our experiment.
6. **2304.03153** Wang & Lim — Zero-Shot Next-Item Recommendation using Large
   Pretrained Language Models (2023, 93 citations). NIR prompt strategy with
   external candidate generator. Code: AGI-Edgerunners/LLM-Next-Item-Rec.
7. **2306.05817** Lin et al. — How Can Recommender Systems Benefit from Large
   Language Models: A Survey (TOIS 2024, 363 citations). Where/how LLMs fit
   into the rec pipeline; covers efficiency, effectiveness, ethics challenges.
8. **2503.05493** RecBench — Can LLMs Outshine Conventional Recommenders? A
   Comparative Evaluation (NeurIPS D&B 2025). Tests 17 LLMs on five domains
   including music; reports up to +5% AUC (CTR) and +170% NDCG@10 (SeqRec)
   over conventional methods.

## Music-specific LLM recommendation

9. **2511.16478** Epure et al. — Music Recommendation with Large Language
   Models: Challenges, Opportunities, and Evaluation (ACM, Nov 2025). 36-page
   survey + the evaluation framework we are adopting (G1–G6 success
   dimensions plus risk diagnostics for hallucination, popularity bias,
   profile hazards, evaluator bias).
10. **2507.16708** Sguerra et al. — Biases in LLM-Generated Musical Taste
    Profiles for Recommendation (RecSys 2025). The most directly relevant
    paper. Generates NL profiles from listening histories with three LLMs
    (Llama 3.2, DeepSeek-R1, Gemini 2.0), runs a 64-user study + downstream
    Recall@10/NDCG@10. Finds rap content lowers ratings, US-origin tracks
    raise them, and that user ratings correlate only weakly with downstream
    accuracy. Code: deezer/recsys25_llm_biases.
11. **2502.13713** Doh et al. — TalkPlay: Multimodal Music Recommendation with
    Large Language Models (Feb 2025). Reformulates rec as token generation;
    expands LLM vocabulary with audio/lyrics/metadata/tag tokens.
12. **2509.09685** TalkPlayData 2 — Agentic synthetic data pipeline for
    multimodal conversational music recommendation (Sep 2025).
13. **2510.01698** TalkPlay-Tools — Conversational Music Recommendation with
    LLM Tool Calling (Oct 2025).
14. **2503.24193** Palumbo et al. (Spotify Research) — Text2Tracks: Prompt-
    based Music Recommendation via Generative Retrieval (2025). Industry
    benchmark; semantic IDs beat title strings by 127%. Code:
    mayurbhangale/text2tracks.
15. **2502.15229** Yun et al. — User Experience with LLM-powered Conversational
    Recommendation Systems: A Case of Music Recommendation (CHI 2025). 3-week
    diary study with 12 participants on custom GPT music CRS.
16. **2508.11671** Boadana et al. — LLM-Based Intelligent Agents for Music
    Recommendation: A Comparison with Classical Content-Based Filtering (2025,
    in Portuguese). Real Spotify export from 19 users over 13 months;
    multi-agent CrewAI architecture with Gemini 2.0 Flash + LLaMA 3.3-70B vs
    TF-IDF cosine CBF. Headline numbers: LLaMA 89.32% Like Rate vs CBF 61%,
    but LLaMA novelty only 11.85% vs CBF 58.5%. Tradeoff:
    satisfaction-vs-discovery, plus 50–60× latency penalty.
17. **2306.09327** McKee et al. — Language-Guided Music Recommendation for
    Video via Prompt Analogies (2023, 33 citations). Useful as a
    cross-modal baseline.
18. **2507.15826** Just Ask for Music (JAM) — Multimodal and Personalized
    Natural Language Music Recommendation (2025).

## Domain context

19. **2409.09378** Prevailing Research Areas for Music AI in the Era of
    Foundation Models (2024). Survey of music-AI research themes; useful for
    framing related work.

## How to read these efficiently

- **Already deeply read** (chunked + in literature review): 2511.16478,
  2507.16708, 2307.14225, 2305.08845, 2508.11671.
- **Skim abstracts via paper-finder JSONL** (already done) for the rest;
  re-chunk if the experiment runner needs deeper detail.

The chunked PDF pages live in `papers/pages/` (one PDF per chunk, ~4 pages
each); see the per-paper `_manifest.txt` files there for the page ranges.
