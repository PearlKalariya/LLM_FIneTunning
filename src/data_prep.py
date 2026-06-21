"""Week-1 deliverable: curate → normalize → stratified split → JSONL.

Run:  python -m src.data_prep
Output: data/processed/{train,val,test}.jsonl  (100% schema-valid)
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from .config import DATA, DATA_PROCESSED, DATA_RAW, SEED
from .schema import INSTRUCTION, parse_and_validate


def to_example(text: str, sentiment: str, key_entities: list[str], summary: str) -> dict:
    """Build one single-turn instruction example with JSON target."""
    target = {"sentiment": sentiment, "key_entities": key_entities, "summary": summary}
    return {
        "instruction": INSTRUCTION,
        "input": text.strip(),
        "output": json.dumps(target, ensure_ascii=False),
        "sentiment": sentiment,  # kept top-level for stratification
    }


def load_local_labeled() -> list[dict]:
    """Load hand-labeled Indian-market set (the differentiator). JSONL of raw labels."""
    path = DATA_RAW / DATA.local_labeled
    if not path.exists():
        print(f"[warn] no local labeled file at {path} — skipping")
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        out.append(to_example(
            row["text"], row["sentiment"],
            row.get("key_entities", []), row["summary"],
        ))
    return out


def _summarize(text: str, max_words: int = 18) -> str:
    """Cheap one-line summary for sentiment-only sources: trim to N words."""
    words = text.split()
    s = " ".join(words[:max_words])
    return s if s.endswith(".") else s + "."


def load_hf_sources(per_source: int = 600) -> list[dict]:
    """Map HF sentiment sources to schema. No-op if `datasets` missing.

    These sources are sentiment-only: key_entities left empty, summary derived
    by trimming. Tradeoff documented in DATASET.md — the hand-labeled Indian set
    carries the entity/summary supervision.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[warn] `datasets` not installed — skipping HF sources")
        return []

    out: list[dict] = []

    # financial_phrasebank: label 0=negative, 1=neutral, 2=positive
    try:
        fpb = load_dataset("financial_phrasebank", "sentences_50agree",
                           split="train")
        fpb_map = {0: "negative", 1: "neutral", 2: "positive"}
        for row in fpb.shuffle(seed=SEED).select(range(min(per_source, len(fpb)))):
            text = row["sentence"]
            out.append(to_example(text, fpb_map[row["label"]], [], _summarize(text)))
    except Exception as e:  # dataset fetch is best-effort
        print(f"[warn] financial_phrasebank skipped: {e}")

    # twitter-financial-news-sentiment: 0=Bearish, 1=Bullish, 2=Neutral
    try:
        tw = load_dataset("zeroshot/twitter-financial-news-sentiment",
                          split="train")
        tw_map = {0: "negative", 1: "positive", 2: "neutral"}
        for row in tw.shuffle(seed=SEED).select(range(min(per_source, len(tw)))):
            text = row["text"]
            out.append(to_example(text, tw_map[row["label"]], [], _summarize(text)))
    except Exception as e:
        print(f"[warn] twitter-financial-news-sentiment skipped: {e}")

    print(f"[info] HF sources -> {len(out)} examples")
    return out


def validate_all(rows: list[dict]) -> list[dict]:
    """Drop any example whose target is not schema-valid. Enforce 100% validity."""
    good = []
    for r in rows:
        res = parse_and_validate(r["output"])
        if res.ok:
            good.append(r)
        else:
            print(f"[drop] invalid target: {res.error}")
    return good


def dedup(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        key = r["input"].lower().strip()
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def stratified_split(rows: list[dict]) -> dict[str, list[dict]]:
    """80/10/10 train/val/test, stratified by sentiment, seeded."""
    rng = random.Random(SEED)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_label[r["sentiment"]].append(r)

    train, val, test = [], [], []
    tr, vr, _ = DATA.splits
    for label, items in by_label.items():
        rng.shuffle(items)
        n = len(items)
        n_tr, n_vr = int(n * tr), int(n * vr)
        train += items[:n_tr]
        val += items[n_tr:n_tr + n_vr]
        test += items[n_tr + n_vr:]
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return {"train": train, "val": val, "test": test}


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    rows = load_local_labeled() + load_hf_sources()
    print(f"[info] loaded {len(rows)} raw examples")
    rows = validate_all(rows)
    rows = dedup(rows)
    print(f"[info] {len(rows)} valid, deduped")

    if not rows:
        print("[error] no examples — add data/raw/indian_market.jsonl or wire HF sources")
        return

    splits = stratified_split(rows)
    for name, data in splits.items():
        out = DATA_PROCESSED / f"{name}.jsonl"
        write_jsonl(data, out)
        labels = defaultdict(int)
        for r in data:
            labels[r["sentiment"]] += 1
        print(f"[ok] {name}: {len(data)} -> {out}  {dict(labels)}")


if __name__ == "__main__":
    main()
