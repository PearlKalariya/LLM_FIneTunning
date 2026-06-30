# LLM Fine-Tuning Platform

Fine-tune `Llama-3.2-3B-Instruct` with QLoRA for **structured extraction from Indian financial news / earnings snippets** — sentiment, key entities, and a one-line summary, emitted as strict JSON. Includes a quantitative baseline-vs-finetuned eval harness and a multi-adapter FastAPI serving layer (hot-swappable LoRA adapters on one base model).

This is the training-loop project: dataset curation → QLoRA fine-tune → before/after eval → serving. It plugs the fine-tuned model into the existing Financial AI Agent as a pluggable extraction component.

## One-line pitch

> Fine-tuned Llama-3.2-3B via QLoRA on a custom Indian financial-news dataset, lifting entity-extraction exact-match from 3.8% (zero-shot baseline) to 87.2%; built a multi-adapter FastAPI serving layer with a W&B-tracked training pipeline.

## Quick links

| Doc | What |
|---|---|
| [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) | Functional + non-functional requirements |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design, data flow, components |
| [docs/DATASET.md](docs/DATASET.md) | Sources, schema, labeling, splits |
| [docs/PROCESSES.md](docs/PROCESSES.md) | Dev workflow, training runbook, checkpointing |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Metrics, harness, baseline-vs-finetuned table |
| [docs/SERVING.md](docs/SERVING.md) | FastAPI API, adapter registry, hot-swap |
| [docs/SECURITY.md](docs/SECURITY.md) | Threat model, secrets, data handling |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Hosting, GGUF export, environments |
| [docs/ROADMAP.md](docs/ROADMAP.md) | 4-week milestone plan + risks |

## Stack

Training: `transformers`, `peft`, `bitsandbytes`, `trl`, `datasets`. Tracking: Weights & Biases. Serving: FastAPI + `peft` adapter hot-swap. Demo: Next.js. Compute: Colab/Kaggle free-tier T4 (QLoRA, 4-bit).

## Run

```bash
make install   # deps (Colab/T4 for training; CPU ok for data/test)
make data      # curate -> stratified split -> data/processed/*.jsonl
make train     # QLoRA fine-tune (run on T4)
make eval      # baseline vs fine-tuned -> eval_results.json
make serve     # FastAPI on :8020
make test      # unit tests (schema + data prep)
```

## Status

Pipeline run end-to-end on a Colab T4: QLoRA fine-tune complete, baseline-vs-finetuned
eval done (entity exact-match 3.8% → 87.2%; see [docs/EVALUATION.md](docs/EVALUATION.md)).
Data prep + unit tests run on CPU (verified, 11 passing). Remaining: serve-time
latency/VRAM numbers and the LLM-judge metric (needs Anthropic credits).
Code follows the [roadmap](docs/ROADMAP.md).
