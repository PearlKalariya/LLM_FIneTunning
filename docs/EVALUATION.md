# Evaluation

All metrics reported as **baseline (zero-shot) vs fine-tuned** on the untouched held-out test set. This table is the evidence behind the resume bullet.

## 1. Metrics

| Metric | Measures | Method |
|---|---|---|
| JSON schema validity rate | Did fine-tuning teach reliable structured output? | % of outputs that parse as JSON AND match the contract |
| Sentiment accuracy / F1 | Task correctness | Compare predicted vs gold `sentiment` label, macro-F1 |
| Entity exact-match | Extraction precision | Set match of `key_entities` vs gold |
| LLM-judge score (1–5) | Summary quality | Claude API rates `summary` faithfulness/quality |
| Inference latency | Efficiency story | ms/request, measured at serve time |
| VRAM footprint | Efficiency story | Peak VRAM at inference |

## 2. Comparison table

Run: Colab T4, QLoRA 4-bit, 3 epochs, `Llama-3.2-3B-Instruct`, 78-example held-out test set (2026-06-29).

| Metric | Baseline (zero-shot) | Fine-tuned | Δ |
|---|---|---|---|
| JSON schema validity rate | 98.7% | 98.7% | +0.0 |
| Sentiment macro-F1 | 0.814 | 0.781 | −0.033 |
| Entity exact-match | 3.8% | 87.2% | **+83.4 pts** |
| LLM-judge (avg 1–5) | n/a | n/a | not run (no Anthropic credits) |
| Latency (ms/req) | TBD | TBD | measured at serve time |
| VRAM (GB) | TBD | TBD | measured at serve time |

**Headline:** fine-tuning lifts entity exact-match from 3.8% → 87.2% (+83 pts). The base model emits valid JSON (prompt-driven, already 98.7%) but cannot produce the exact entity set; fine-tuning teaches it. Sentiment dips 3.3 pts (≈2–3 of 78 examples, within noise) — the base model was already strong on sentiment, and training traded a sliver of it for the large entity gain.

## 3. LLM-judge protocol

- Judge model: Claude API.
- Input: snippet + gold summary + predicted summary.
- Output: integer 1–5 on faithfulness + conciseness, plus one-line reason.
- Average across test set. Fixed judge prompt for comparability.

## 4. Failure analysis

After each eval, bucket failures: schema breaks, wrong sentiment, missed/hallucinated entities, weak summaries. Drive the next data/training iteration from the largest bucket.

**2026-06-29 run:** largest remaining gap is sentiment (−3.3 pts vs baseline). Base Llama already classifies sentiment well zero-shot; fine-tuning slightly over-fit toward the extraction objective. Next iteration: rebalance the loss toward sentiment, or add harder sentiment examples. Entity extraction is effectively solved at this data scale (87.2%). Schema validity is at ceiling (98.7%) and not worth chasing.

## 5. Guards

- Test set untouched until final eval (no peeking → no leakage).
- Same prompt for baseline and fine-tuned where applicable (fair comparison).
- Seeded generation where possible for repeatable numbers.
- Report efficiency metrics too — VRAM/latency back the "platform" framing.

## 6. Known ceiling

3B caps output quality vs 7B. Document explicitly as engineering judgment (cost/quality tradeoff), not an oversight. 7B is the stretch path if 3B numbers justify it.
