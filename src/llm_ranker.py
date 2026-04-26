"""LLM zero-shot ranker following the Hou et al. (2023) sequential prompt
template, adapted to a Spotify-export-style listening history.

Two modes:
- ranker (default): given user history + 20 candidates, return a ranked list
  of candidate indices.
- ranker_no_history (ablation): given only the 20 candidates, return a ranked
  list "by general musical merit" — the no-personalization floor for the LLM.

For Experiment 3 there is also a free_generation mode that asks the LLM to
recommend track names without a candidate pool, used to measure the
hallucination rate (entities-resolved-share).
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import List, Optional

from openai import OpenAI

CLAUDE_MODEL = "anthropic/claude-sonnet-4.5"
GPT_MODEL = "gpt-4.1"


def _client_for(model: str) -> OpenAI:
    if model.startswith("anthropic/") or model.startswith("openrouter/"):
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_KEY"),
        )
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _format_history(history_top: list, max_tracks: int = 50) -> str:
    lines = []
    for i, h in enumerate(history_top[:max_tracks], 1):
        lines.append(f"{i}. {h['artist_name']} - {h['track_name']}")
    return "\n".join(lines)


def _format_candidates(candidates: list) -> str:
    lines = []
    for i, c in enumerate(candidates):
        lines.append(f"[{i}] {c['artists']} - {c['track_name']}")
    return "\n".join(lines)


RANKING_PROMPT = """You are a music recommendation expert. The user below has
exported their listening history from Spotify. Your task is to rank a list of
candidate tracks by how likely the user is to enjoy them, based on their
listening history.

# User's listening history (most-played tracks first)
{history}

# Candidate tracks
{candidates}

# Instructions
Rank the candidate tracks from most likely to least likely to be enjoyed by
this user.

Output format: a single line with the candidate indices separated by spaces,
ordered from BEST first to WORST last. Do not include any other text.
Example output: 3 7 1 0 12 5 ...

Your ranking:"""


NO_HISTORY_PROMPT = """You are a music expert. Below is a list of candidate
tracks. Rank them by general musical quality and broad appeal — without any
information about a specific listener.

# Candidate tracks
{candidates}

# Instructions
Rank the candidate tracks by general musical merit, from best (most likely to
appeal broadly) to worst.

Output format: a single line with the candidate indices separated by spaces,
ordered from BEST first to WORST last. Do not include any other text.
Example output: 3 7 1 0 12 5 ...

Your ranking:"""


FREE_GEN_PROMPT = """You are a music recommendation expert. The user below has
exported their listening history from Spotify. Recommend 10 tracks the user
might enjoy that are NOT already in their listening history.

# User's listening history (most-played tracks first)
{history}

# Instructions
Output exactly 10 recommendations as a JSON array. Each item must be an
object with two string fields: "artist" and "track". Do not include any other
text outside of the JSON array.

Example output:
[{{"artist":"Radiohead","track":"Idioteque"}},{{"artist":"Bjork","track":"Hyperballad"}}, ...]

Your recommendations:"""


def _parse_ranking(text: str, n_candidates: int) -> Optional[List[int]]:
    """Parse a ranked list of indices from LLM output. Returns a permutation
    of range(n_candidates), or None if parsing fails."""
    if not text:
        return None
    # Find the longest run of integers in the response.
    # First try the line with the most digits.
    candidates_lines = [ln for ln in text.splitlines() if re.search(r"\d", ln)]
    candidates_lines.append(text)  # fallback to whole text
    best_perm = None
    best_count = -1
    for line in candidates_lines:
        ints = [int(x) for x in re.findall(r"\b\d+\b", line)]
        # keep only valid indices, dedup preserving order
        seen = set()
        perm = []
        for x in ints:
            if 0 <= x < n_candidates and x not in seen:
                seen.add(x)
                perm.append(x)
        if len(perm) > best_count:
            best_count = len(perm)
            best_perm = perm
    if best_perm is None or len(best_perm) == 0:
        return None
    # Append any missing indices at the end (so we always return a full perm).
    missing = [i for i in range(n_candidates) if i not in set(best_perm)]
    return best_perm + missing


def llm_rank(
    pool: dict,
    model: str = CLAUDE_MODEL,
    use_history: bool = True,
    temperature: float = 0.0,
    max_tokens: int = 200,
    max_retries: int = 3,
) -> dict:
    """Run the LLM ranker on a pool. Returns dict with keys:
    order (list[int]), latency_s, tokens_in, tokens_out, raw, parsed_ok."""
    client = _client_for(model)
    if use_history:
        prompt = RANKING_PROMPT.format(
            history=_format_history(pool["history_top"]),
            candidates=_format_candidates(pool["candidates"]),
        )
    else:
        prompt = NO_HISTORY_PROMPT.format(
            candidates=_format_candidates(pool["candidates"]),
        )
    last_err = None
    t0 = time.time()
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = r.choices[0].message.content or ""
            usage = r.usage
            tokens_in = usage.prompt_tokens if usage else 0
            tokens_out = usage.completion_tokens if usage else 0
            order = _parse_ranking(text, len(pool["candidates"]))
            parsed_ok = order is not None and len(order) == len(pool["candidates"])
            if not parsed_ok:
                # fallback: range order
                order = list(range(len(pool["candidates"])))
            return {
                "order": order,
                "latency_s": time.time() - t0,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "raw": text,
                "parsed_ok": parsed_ok,
                "model": model,
                "use_history": use_history,
            }
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"LLM ranking failed after {max_retries} retries: {last_err}")


def llm_free_generate(
    pool: dict,
    model: str = CLAUDE_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 600,
    max_retries: int = 3,
) -> dict:
    """Free-generation mode: ask LLM to produce 10 (artist, track) recs."""
    client = _client_for(model)
    prompt = FREE_GEN_PROMPT.format(
        history=_format_history(pool["history_top"]),
    )
    last_err = None
    t0 = time.time()
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = r.choices[0].message.content or ""
            # Parse JSON array
            recs = []
            try:
                # Find outermost [...]
                m = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
                if m:
                    recs = json.loads(m.group(0))
            except Exception:
                recs = []
            usage = r.usage
            return {
                "recs": recs,
                "latency_s": time.time() - t0,
                "tokens_in": usage.prompt_tokens if usage else 0,
                "tokens_out": usage.completion_tokens if usage else 0,
                "raw": text,
                "model": model,
            }
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"LLM free-gen failed after {max_retries} retries: {last_err}")


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
    print(f"User {uid}: positives at {pool['positive_indices']}")
    res = llm_rank(pool, model=CLAUDE_MODEL)
    print(f"Claude: parsed_ok={res['parsed_ok']}, order={res['order']}")
    print(f"  tokens in/out = {res['tokens_in']}/{res['tokens_out']}, latency = {res['latency_s']:.1f}s")
    print(f"  raw: {res['raw'][:200]}")
