# Requirements

## 1. Goal

Train and serve a fine-tuned `Llama-3.2-3B-Instruct` that extracts structured JSON (`sentiment`, `key_entities`, `summary`) from Indian financial news / earnings snippets, with measurable improvement over the zero-shot baseline.

## 2. Functional requirements

| ID | Requirement |
|---|---|
| FR-1 | Ingest financial-sentiment data from HF (`financial_phrasebank`, `zeroshot/twitter-financial-news-sentiment`) plus hand-labeled Indian-market set. |
| FR-2 | Convert all examples to single-turn instruction format with JSON target. |
| FR-3 | Stratified 80/10/10 train/val/test split by sentiment label. |
| FR-4 | QLoRA fine-tune (4-bit NF4 + LoRA) producing a named adapter checkpoint. |
| FR-5 | Eval harness scores baseline vs fine-tuned on held-out test set across all metrics in [EVALUATION.md](EVALUATION.md). |
| FR-6 | Adapter registry stores multiple named LoRA adapters, loadable at inference. |
| FR-7 | `POST /infer {task, text}` returns schema-valid structured JSON. |
| FR-8 | Adapter hot-swap per request via `peft set_adapter()` — no base-model reload. |
| FR-9 | Training runs tracked in W&B (loss curves, run comparison, config). |
| FR-10 | (Optional) Next.js demo: side-by-side base vs fine-tuned output. |

## 3. Output schema (contract)

```json
{
  "sentiment": "positive | negative | neutral",
  "key_entities": ["string", "..."],
  "summary": "string (one line)"
}
```

Invalid JSON or off-schema output counts as a failure in eval (see JSON schema validity rate).

## 4. Non-functional requirements

| ID | Requirement | Target |
|---|---|---|
| NFR-1 | Fits free-tier 16GB VRAM (T4) for QLoRA training | 3B @ 4-bit |
| NFR-2 | Zero training cost | Colab/Kaggle free tier |
| NFR-3 | Inference latency reported per request | measured, in eval table |
| NFR-4 | VRAM footprint at serve time reported | measured |
| NFR-5 | Reproducible runs | seeded, config tracked in W&B |
| NFR-6 | Resumable training | checkpoint every N steps |

## 5. In scope

Dataset curation, QLoRA training pipeline, eval harness, multi-adapter FastAPI server, W&B tracking, optional demo UI, optional GGUF/llama.cpp export.

## 6. Out of scope (v1)

- Full fine-tune (non-LoRA) — QLoRA only.
- 7B model — stretch goal, gated on 3B results justifying it.
- Multi-turn / chat extraction — single-turn only.
- RLHF / DPO — SFT only.
- Production-grade auth/multi-tenant serving — demo-grade.

## 7. Assumptions

- Free-tier T4 GPU available (Colab/Kaggle 30 GPU-hrs/week).
- Hand-labeling ~150–300 Indian-market examples is acceptable manual effort.
- Total dataset 800–1,500 examples is sufficient for this narrow task.
- Claude API key available for LLM-judge eval metric.
