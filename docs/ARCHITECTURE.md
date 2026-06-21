# Architecture

## 1. Pipeline

```
Raw data (HF datasets + hand-curated Indian-market set)
        │
        ▼
  Dataset prep  (instruction → JSON target, stratified split)
        │
        ▼
  QLoRA fine-tune  (peft + bitsandbytes + trl SFTTrainer)
        │   └── W&B run tracking (loss, config, comparison)
        ▼
  Eval harness  (baseline zero-shot vs fine-tuned, held-out test)
        │
        ▼
  Adapter registry  (named LoRA adapters, metadata)
        │
        ▼
  FastAPI serving  (base loaded once; set_adapter() per request)
        │
        ▼
  Demo UI (Next.js, optional)  — side-by-side base vs fine-tuned
```

## 2. Components

### Dataset prep
Loads HF sources + local labeled JSONL, normalizes to one instruction schema, deduplicates, stratifies by sentiment, writes `train/val/test` JSONL. Deterministic (seeded).

### Trainer
QLoRA via `trl` `SFTTrainer`. Base loaded 4-bit NF4, LoRA adapters on attention projections. Checkpoints to disk every N steps for Colab-timeout resume. Logs to W&B.

### Eval harness
Runs both base (zero-shot, prompt-only) and fine-tuned (base + adapter) over the test set. Computes schema validity, sentiment F1, entity exact-match, LLM-judge score, latency, VRAM. Emits a comparison table (markdown + JSON).

### Adapter registry
Directory of adapter checkpoints + `registry.json` mapping `task_name → {adapter_path, base_model, metrics, created_at}`. The serving layer reads this to know what it can load.

### Serving layer
FastAPI. Base model loaded once into memory. Adapters attached/activated via `peft` `set_adapter()` per request — this multi-adapter-on-one-base design is what makes it a *platform*, not a single checkpoint. `POST /infer {task, text}` → parsed, schema-validated JSON.

## 3. Key design decisions

| Decision | Choice | Why |
|---|---|---|
| Method | QLoRA (4-bit NF4 + LoRA) | Fits free T4; resume-recognizable; cheap |
| Base model | Llama-3.2-3B-Instruct | 3B fits 16GB VRAM at 4-bit; 7B is stretch |
| LoRA targets | `q_proj,k_proj,v_proj,o_proj` | Attention-only; standard, light |
| Serving | One base + swappable adapters | Platform framing; memory-efficient |
| Output | Strict JSON contract | Measurable, integrates with Financial Agent |

## 4. LoRA / quant config

```python
# peft LoraConfig
r=16, lora_alpha=32, lora_dropout=0.05
target_modules=["q_proj","k_proj","v_proj","o_proj"]

# bitsandbytes
load_in_4bit=True, bnb_4bit_quant_type="nf4",
bnb_4bit_compute_dtype=torch.bfloat16

# trl SFTTrainer
epochs=3, lr=2e-4, lr_scheduler="cosine",
per_device_train_batch_size=4, gradient_accumulation_steps=4
```

## 5. Integration

Fine-tuned adapter becomes a pluggable extraction component for the existing **Financial AI Agent** project — connected two-project narrative, not two unrelated ones.
