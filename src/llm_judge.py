"""LLM-judge summary quality (1–5) via Claude API. Fixed prompt for comparability."""
from __future__ import annotations

import os
import re

JUDGE_MODEL = "claude-opus-4-8"

JUDGE_PROMPT = """You rate the quality of a one-line summary of a financial snippet.

Snippet:
{snippet}

Reference summary:
{gold}

Candidate summary:
{pred}

Rate the candidate 1-5 on faithfulness (no hallucination, matches snippet) and
conciseness (one factual line). Reply with ONLY the integer 1-5."""


def score_one(snippet: str, gold: str, pred: str) -> int:
    """Single judgment. Returns 1–5; 1 on API/parse failure (conservative)."""
    try:
        import anthropic
    except ImportError:
        return 0
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return 0
    client = anthropic.Anthropic(api_key=key)
    msg = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=8,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            snippet=snippet, gold=gold, pred=pred)}],
    )
    text = msg.content[0].text.strip()
    m = re.search(r"[1-5]", text)
    return int(m.group()) if m else 1


def average_score(snippets, golds, preds) -> float:
    """Mean 1–5 over the set. 0.0 if judge unavailable (no key / no sdk)."""
    scores = [
        score_one(s, g, p)
        for s, g, p in zip(snippets, golds, preds)
        if p is not None
    ]
    scores = [x for x in scores if x > 0]
    return sum(scores) / len(scores) if scores else 0.0
