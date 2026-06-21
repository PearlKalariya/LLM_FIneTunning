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

## 2. Comparison table (fill after run)

| Metric | Baseline (zero-shot) | Fine-tuned | Δ |
|---|---|---|---|
| JSON schema validity rate | __% | __% | |
| Sentiment macro-F1 | __ | __ | |
| Entity exact-match | __% | __% | |
| LLM-judge (avg 1–5) | __ | __ | |
| Latency (ms/req) | __ | __ | |
| VRAM (GB) | __ | __ | |

## 3. LLM-judge protocol

- Judge model: Claude API.
- Input: snippet + gold summary + predicted summary.
- Output: integer 1–5 on faithfulness + conciseness, plus one-line reason.
- Average across test set. Fixed judge prompt for comparability.

## 4. Failure analysis

After each eval, bucket failures: schema breaks, wrong sentiment, missed/hallucinated entities, weak summaries. Drive the next data/training iteration from the largest bucket.

## 5. Guards

- Test set untouched until final eval (no peeking → no leakage).
- Same prompt for baseline and fine-tuned where applicable (fair comparison).
- Seeded generation where possible for repeatable numbers.
- Report efficiency metrics too — VRAM/latency back the "platform" framing.

## 6. Known ceiling

3B caps output quality vs 7B. Document explicitly as engineering judgment (cost/quality tradeoff), not an oversight. 7B is the stretch path if 3B numbers justify it.
