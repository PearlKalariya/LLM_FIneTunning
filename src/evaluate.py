"""Eval harness (Week 3): baseline vs fine-tuned on held-out test set.

Run:  python -m src.evaluate --task financial_extraction
Emits markdown + JSON comparison table.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from .config import DATA_PROCESSED, ROOT
from .schema import parse_and_validate


def load_test() -> list[dict]:
    path = DATA_PROCESSED / "test.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def schema_validity(preds: list[str]) -> float:
    ok = sum(parse_and_validate(p).ok for p in preds)
    return ok / len(preds) if preds else 0.0


def sentiment_f1(preds: list[dict | None], golds: list[dict]) -> float:
    """Macro-F1 over sentiment classes. Invalid pred = wrong."""
    from sklearn.metrics import f1_score
    y_true, y_pred = [], []
    for p, g in zip(preds, golds):
        y_true.append(g["sentiment"])
        y_pred.append(p["sentiment"] if p else "__invalid__")
    return f1_score(y_true, y_pred, average="macro", labels=sorted(set(y_true)))


def entity_exact_match(preds: list[dict | None], golds: list[dict]) -> float:
    hits = 0
    for p, g in zip(preds, golds):
        if p and set(p["key_entities"]) == set(g["key_entities"]):
            hits += 1
    return hits / len(golds) if golds else 0.0


def llm_judge(examples, preds: list[dict | None], golds: list[dict]) -> float:
    """Avg 1–5 summary quality via Claude. 0.0 if no ANTHROPIC_API_KEY."""
    from .llm_judge import average_score
    snippets = [e["input"] for e in examples]
    gold_sum = [g["summary"] for g in golds]
    pred_sum = [p["summary"] if p else None for p in preds]
    return average_score(snippets, gold_sum, pred_sum)


def run_model(examples, predict) -> tuple[list[str], list[dict | None]]:
    raw, parsed = [], []
    for e in examples:
        out = predict(e)
        raw.append(out)
        res = parse_and_validate(out)
        parsed.append(res.data if res.ok else None)
    return raw, parsed


def evaluate(predict, examples) -> dict:
    golds = [parse_and_validate(e["output"]).data for e in examples]
    raw, parsed = run_model(examples, predict)
    return {
        "schema_validity": schema_validity(raw),
        "sentiment_f1": sentiment_f1(parsed, golds),
        "entity_exact_match": entity_exact_match(parsed, golds),
        "llm_judge": llm_judge(examples, parsed, golds),
    }


def to_markdown(baseline: dict, finetuned: dict) -> str:
    rows = [
        ("JSON schema validity", "schema_validity"),
        ("Sentiment macro-F1", "sentiment_f1"),
        ("Entity exact-match", "entity_exact_match"),
        ("LLM-judge (1–5)", "llm_judge"),
    ]
    lines = ["| Metric | Baseline | Fine-tuned | Δ |", "|---|---|---|---|"]
    for label, key in rows:
        b, f = baseline[key], finetuned[key]
        lines.append(f"| {label} | {b:.3f} | {f:.3f} | {f - b:+.3f} |")
    return "\n".join(lines)


def _write_metrics(task: str, metrics: dict) -> None:
    """Backfill fine-tuned metrics into the adapter registry."""
    from .config import ADAPTERS_DIR
    reg_path = ADAPTERS_DIR / "registry.json"
    if not reg_path.exists():
        return
    reg = json.loads(reg_path.read_text())
    if task in reg:
        reg[task]["metrics"] = {k: round(v, 4) for k, v in metrics.items()}
        reg_path.write_text(json.dumps(reg, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="financial_extraction")
    ap.add_argument("--cpu", action="store_true",
                    help="load full precision (no 4-bit) — debug only")
    args = ap.parse_args()
    examples = load_test()

    from .inference import get_engine
    eng = get_engine(load_4bit=not args.cpu)

    def baseline_predict(e): return eng.generate(e["input"], task=None)
    def finetuned_predict(e): return eng.generate(e["input"], task=args.task)

    base = evaluate(baseline_predict, examples)
    fine = evaluate(finetuned_predict, examples)
    _write_metrics(args.task, fine)

    table = to_markdown(base, fine)
    print(table)
    out = ROOT / "eval_results.json"
    out.write_text(json.dumps({"baseline": base, "finetuned": fine}, indent=2))
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()
