# Processes

## 1. Dev workflow

1. Branch from `main` per change.
2. Dataset/config changes are version-controlled (they define a run).
3. Every training run logs to W&B with the exact config — runs are reproducible from the dashboard.
4. Eval table regenerated after every fine-tune; numbers go in the README bullet.

## 2. Environment setup

- Python env with `transformers, peft, bitsandbytes, trl, datasets, wandb, fastapi, uvicorn`.
- Secrets via env vars / `.env` (gitignored): `WANDB_API_KEY`, `HF_TOKEN`, `ANTHROPIC_API_KEY`. See [SECURITY.md](SECURITY.md).
- Training runs on Colab/Kaggle T4; serving runs locally or on a serverless GPU.

## 3. Training runbook

1. **Prep data** → `data/processed/{train,val,test}.jsonl`. Verify schema validity 100%.
2. **Load base** Llama-3.2-3B-Instruct 4-bit NF4.
3. **Attach LoRA** (`r=16, alpha=32, dropout=0.05`, attention projections).
4. **Train** `trl SFTTrainer`: 3 epochs, lr 2e-4, cosine, batch 4 × grad-accum 4.
5. **Checkpoint every N steps** to disk (Colab-timeout resume).
6. **Watch W&B**: train + val loss. Early-stop on val-loss divergence.
7. **Save adapter** to registry with metadata (config, metrics, timestamp).

## 4. Checkpoint / resume (Colab timeout guard)

- `save_steps=N`, keep last K checkpoints.
- On session death: re-mount, reload last checkpoint, resume trainer state.
- Never lose more than N steps of progress.

## 5. Eval process

1. Run baseline (zero-shot, prompt only) over test set → record metrics.
2. Run fine-tuned (base + adapter) over same test set → record metrics.
3. Generate baseline-vs-finetuned comparison table (markdown + JSON).
4. Inspect failure cases; feed fixes into next data/training iteration.
5. Lock final numbers into README + resume bullet.

See [EVALUATION.md](EVALUATION.md) for metric definitions.

## 6. Iteration loop

```
train → eval → inspect failures → fix data/config → retrain
```

Week-3 plan: a second training run with fixes from the first eval pass.

## 7. Serving process

1. Serving layer reads adapter `registry.json`.
2. Base model loaded once at startup.
3. Per request: `set_adapter(task)`, run inference, parse + schema-validate JSON, return.
4. On invalid output: one constrained retry, else error with reason.

See [SERVING.md](SERVING.md).

## 8. Release checklist

- [ ] Eval table populated with real numbers.
- [ ] W&B dashboard public/linkable.
- [ ] Adapter checkpoint(s) in registry with metadata.
- [ ] FastAPI server runs, `/infer` returns schema-valid JSON.
- [ ] README bullet filled with real X% → Y%.
- [ ] Secrets verified gitignored ([SECURITY.md](SECURITY.md)).
