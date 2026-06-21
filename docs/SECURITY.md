# Security

Demo-grade project, but treat secrets and data correctly — this is interview-visible code.

## 1. Secrets

| Secret | Use | Handling |
|---|---|---|
| `HF_TOKEN` | Pull gated models/datasets | env var / `.env`, gitignored |
| `WANDB_API_KEY` | Training tracking | env var / `.env`, gitignored |
| `ANTHROPIC_API_KEY` | LLM-judge eval | env var / `.env`, gitignored |

Rules:
- Never commit keys. `.env` and `*.key` in `.gitignore`.
- No secrets in notebooks, W&B configs, or logs.
- Rotate any key that touches a public Colab/Kaggle notebook after use.
- Colab/Kaggle: inject via notebook secrets, not inline cells.

## 2. Data handling

- Source data is public financial news / HF datasets — no PII expected.
- Hand-labeled set: verify no private/non-redistributable content before committing.
- Respect HF dataset licenses; record each source's license in [DATASET.md](DATASET.md).
- Do not commit large raw dumps; keep raw data gitignored, commit only the labeled differentiator set.

## 3. Serving layer

- Validate + size-limit all `/infer` input (cap text length).
- Never return unvalidated model output — schema-check first ([SERVING.md](SERVING.md) §5).
- Demo server: no auth by default. Before any public deploy, add an API key / rate limit.
- Treat model output as untrusted: never `eval()`/exec it; parse JSON strictly.

## 4. Model / supply chain

- Pin model + dataset revisions where possible (reproducibility + integrity).
- Pin dependency versions; the QLoRA stack (`bitsandbytes`, `peft`, `trl`) is version-sensitive.
- Pull base model only from the official `meta-llama` HF repo.

## 5. Prompt-injection note

Input snippets are untrusted text. The model may be steered by adversarial snippet content. Mitigation: strict output schema validation gates anything returned; the summary field is the only free-text surface and is never executed.

## 6. Checklist

- [ ] `.env`, `*.key`, `data/raw/` in `.gitignore`.
- [ ] No secrets in committed notebooks/configs/logs.
- [ ] Dataset licenses recorded.
- [ ] `/infer` input validated + length-capped.
- [ ] Output schema-validated before return.
- [ ] Dep + model revisions pinned.
