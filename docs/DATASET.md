# Dataset

## 1. Sources

| Source | Type | Use |
|---|---|---|
| `financial_phrasebank` (HF) | Labeled sentiment | General financial sentiment signal |
| `zeroshot/twitter-financial-news-sentiment` (HF) | Labeled sentiment | More sentiment variety, short-form |
| Hand-curated Indian-market set | Self-labeled | ~150–300 NSE/BSE headlines + earnings snippets — makes the fine-tune *yours* |

## 2. Target schema

Single-turn, instruction-tuned, output is strict JSON:

```json
{
  "sentiment": "positive | negative | neutral",
  "key_entities": ["TCS", "Q3 revenue", "..."],
  "summary": "TCS Q3 revenue rose 7% YoY beating estimates."
}
```

## 3. Instruction format

Each example: a fixed system/instruction prompt asking for the JSON contract, the input snippet, and the JSON target as completion. Keep instruction text identical across examples so the model learns the task, not prompt variance.

## 4. Sizing

| Bucket | Count |
|---|---|
| Total instruction examples | 800–1,500 |
| Hand-labeled Indian-market | 150–300 |
| Remainder | From HF sources, reformatted to schema |

Small on purpose — LoRA on a narrow task rewards task specificity over volume.

## 5. Splits

80 / 10 / 10 train / val / test. **Stratified by sentiment label** so each split holds the class balance. Seeded for reproducibility. **Test set stays untouched** until final eval (overfitting guard).

## 6. Labeling guidelines (Indian-market set)

- `sentiment`: market-impact direction of the snippet, not author tone.
- `key_entities`: tickers, companies, financial metrics, events. Surface forms as they appear.
- `summary`: one line, factual, ≤ ~20 words, no opinion.
- Ambiguous → `neutral`. Log edge cases for consistency review.

## 6a. HF source tradeoff

`financial_phrasebank` and the Twitter set are **sentiment-only** — no entity or
summary labels. On reformat (`load_hf_sources`) `key_entities` is left empty and
`summary` is a trimmed first-line of the snippet. So these sources supervise
sentiment; the **hand-labeled Indian set carries entity + summary supervision**.
Keep the HF fraction balanced against the hand-labeled set so the model still
learns real extraction, not empty-entity shortcuts.

## 7. Cleaning

- Deduplicate near-identical snippets.
- Drop examples that can't map to the schema.
- Normalize whitespace/encoding.
- Validate every target parses as JSON and matches the contract before training.

## 8. Storage

Local JSONL: `data/raw/`, `data/processed/{train,val,test}.jsonl`. Raw HF pulls cached. Hand-labeled file version-controlled (it's the differentiator). See [SECURITY.md](SECURITY.md) for data-handling notes.
