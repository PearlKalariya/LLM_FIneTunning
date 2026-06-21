# Serving

FastAPI layer. One base model in memory, multiple LoRA adapters hot-swapped per request. This multi-adapter-on-one-base design is the "platform" framing.

## 1. Lifecycle

1. Startup: load base Llama-3.2-3B-Instruct once.
2. Read adapter `registry.json`; pre-attach known adapters via `peft`.
3. Per request: `set_adapter(task)`, run inference, parse + validate JSON, return.
4. Invalid output: one constrained retry, then error with reason.

## 2. API

### `POST /infer`

Request:
```json
{ "task": "financial_extraction", "text": "TCS Q3 revenue rose 7% YoY beating estimates." }
```

Response:
```json
{
  "sentiment": "positive",
  "key_entities": ["TCS", "Q3 revenue"],
  "summary": "TCS Q3 revenue rose 7% YoY beating estimates."
}
```

Errors: `400` bad input, `404` unknown task/adapter, `422` model output failed schema validation after retry, `500` inference error.

### `GET /adapters`
List registered adapters + metadata (task, base model, metrics, created_at).

### `GET /health`
Liveness + base-model-loaded + VRAM.

## 3. Adapter registry

`registry.json`:
```json
{
  "financial_extraction": {
    "adapter_path": "adapters/financial_extraction",
    "base_model": "meta-llama/Llama-3.2-3B-Instruct",
    "metrics": { "sentiment_f1": 0.0, "schema_validity": 0.0 },
    "created_at": "2026-01-01T00:00:00Z"
  }
}
```

Adding an adapter = drop checkpoint + add registry entry. No code change.

## 4. Hot-swap

`peft set_adapter()` switches active adapter without reloading the base — fast, memory-cheap. Multiple task-specific adapters share one base in VRAM.

## 5. Output validation

Every model output parsed as JSON and checked against the contract ([REQUIREMENTS.md](REQUIREMENTS.md) §3) before returning. Never return unvalidated model text to the caller.

## 6. Optional CPU path

Merge adapter into base, quantize to GGUF, serve via `llama.cpp` — CPU inference, cheap hosting, no serve-time GPU. See [DEPLOYMENT.md](DEPLOYMENT.md).
