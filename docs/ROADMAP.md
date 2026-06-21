# Roadmap

## Milestone plan (4 weeks)

| Week | Deliverable |
|---|---|
| 1 | Dataset curated + cleaned; baseline (zero-shot) eval numbers locked in |
| 2 | QLoRA training pipeline running; first fine-tune complete; W&B dashboard live |
| 3 | Full eval harness; iterate on failure cases; second training run with fixes |
| 4 | FastAPI multi-adapter server; demo UI; deployment; README + writeup |

## Exit criteria per week

- **W1**: `train/val/test.jsonl` exist, 100% schema-valid; baseline metrics in [EVALUATION.md](EVALUATION.md) table.
- **W2**: adapter checkpoint in registry; W&B run linkable; first fine-tuned eval numbers.
- **W3**: full metric suite running; failure buckets identified; v2 adapter beats v1.
- **W4**: `/infer` live and schema-valid; demo UI side-by-side; README bullet filled with real X%→Y%.

## Stretch goals

- Mistral-7B-Instruct-v0.3 fine-tune (gated on 3B results justifying cost).
- GGUF + llama.cpp CPU serving path.
- Second task adapter (proves multi-adapter platform claim).

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Colab session timeouts | Checkpoint every N steps; resume from checkpoint ([PROCESSES.md](PROCESSES.md) §4) |
| Overfitting on small dataset | Monitor val loss; early-stop; keep test set untouched until final eval |
| 3B ceiling on output quality | Document explicitly as a known 3B-vs-7B tradeoff — reads as engineering judgment |
| QLoRA stack version drift | Pin `bitsandbytes`/`peft`/`trl` versions ([SECURITY.md](SECURITY.md) §4) |
| Weak hand-labels | Labeling guidelines + edge-case log ([DATASET.md](DATASET.md) §6) |

## Resume bullet (fill real numbers post-run)

> Fine-tuned Llama-3.2-3B via QLoRA on a custom Indian financial-news dataset, improving structured-extraction accuracy from X% (zero-shot baseline) to Y%; built a multi-adapter FastAPI serving layer with W&B-tracked training pipeline.
