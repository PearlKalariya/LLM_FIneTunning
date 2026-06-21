"""Output contract + strict validation. Never return unvalidated model text."""
from __future__ import annotations

import json
from dataclasses import dataclass

from .config import SENTIMENTS

INSTRUCTION = (
    "You extract structured data from a financial news or earnings snippet. "
    "Return ONLY a JSON object with keys: "
    '"sentiment" (one of positive, negative, neutral), '
    '"key_entities" (array of strings), '
    '"summary" (one factual line). No prose, no markdown.'
)


@dataclass
class ValidationResult:
    ok: bool
    data: dict | None
    error: str | None = None


def parse_and_validate(text: str) -> ValidationResult:
    """Parse model output as JSON and check it matches the contract."""
    raw = _strip_fences(text).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return ValidationResult(False, None, f"invalid JSON: {e}")

    if not isinstance(data, dict):
        return ValidationResult(False, None, "top level not an object")

    missing = {"sentiment", "key_entities", "summary"} - data.keys()
    if missing:
        return ValidationResult(False, None, f"missing keys: {sorted(missing)}")

    if data["sentiment"] not in SENTIMENTS:
        return ValidationResult(False, None, f"bad sentiment: {data['sentiment']!r}")

    ents = data["key_entities"]
    if not isinstance(ents, list) or not all(isinstance(x, str) for x in ents):
        return ValidationResult(False, None, "key_entities must be list[str]")

    if not isinstance(data["summary"], str) or not data["summary"].strip():
        return ValidationResult(False, None, "summary must be non-empty string")

    return ValidationResult(True, {
        "sentiment": data["sentiment"],
        "key_entities": ents,
        "summary": data["summary"].strip(),
    })


def _strip_fences(text: str) -> str:
    """Drop ```json ... ``` fences if a model wraps output."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        if t.endswith("```"):
            t = t[: -3]
    return t
