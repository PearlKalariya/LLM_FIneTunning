# Deployment

## 1. Environments

| Env | Where | Use |
|---|---|---|
| Training | Colab / Kaggle free T4 | QLoRA runs (30 GPU-hrs/week) |
| Serving (GPU) | Modal / RunPod serverless | Pay-per-second adapter serving |
| Serving (CPU) | HF Spaces / any box | GGUF + llama.cpp, no GPU |
| Demo UI | Vercel / HF Spaces | Next.js side-by-side demo |

## 2. Training (Colab/Kaggle)

- Inject secrets via notebook secrets, not inline ([SECURITY.md](SECURITY.md)).
- Checkpoint every N steps to mounted drive (timeout resume, [PROCESSES.md](PROCESSES.md) §4).
- Push adapter checkpoint + metadata to registry on completion.
- W&B run linkable for the writeup.

## 3. GPU serving (Modal / RunPod)

- Container: base model + adapter registry + FastAPI.
- Base loaded once; adapters hot-swapped ([SERVING.md](SERVING.md)).
- Serverless GPU = pay only per request; scales to zero when idle.

## 4. CPU serving (cheap path)

1. Merge LoRA adapter into base weights.
2. Quantize merged model to GGUF.
3. Serve via `llama.cpp` on CPU.
4. Host on HF Spaces / small VM — no serve-time GPU cost.

Tradeoff: lower throughput/quality vs GPU; fine for a demo.

## 5. Demo UI

Next.js app calls the FastAPI `/infer`. Shows base vs fine-tuned output side by side — the visual proof of the fine-tune's value.

## 6. Config

- All env-specific values via env vars (model id, adapter dir, API base, port).
- No hardcoded secrets or paths.
- Pin dependency + model revisions per [SECURITY.md](SECURITY.md) §4.

## 7. Pre-deploy checklist

- [ ] Secrets in env, not committed.
- [ ] `/infer` input validated + length-capped.
- [ ] Add API key + rate limit before any public endpoint.
- [ ] Registry + adapter checkpoints present.
- [ ] Health check green (base loaded, VRAM ok).
