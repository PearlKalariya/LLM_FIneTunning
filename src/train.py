"""QLoRA fine-tune (Week 2). Stub: structure + config wired, heavy deps lazy.

Run on T4 (Colab/Kaggle):  python -m src.train --task financial_extraction
Checkpoints every TRAIN.save_steps (Colab-timeout resume).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ADAPTERS_DIR, DATA_PROCESSED, LORA, TRAIN, SEED


def build_prompt(example: dict) -> str:
    """Single-turn instruction → JSON target, formatted for SFTTrainer."""
    return (
        f"<|instruction|>\n{example['instruction']}\n"
        f"<|input|>\n{example['input']}\n"
        f"<|output|>\n{example['output']}"
    )


def load_split(name: str) -> list[dict]:
    path = DATA_PROCESSED / f"{name}.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def train(task: str) -> Path:
    # lazy imports — keep module importable without GPU stack installed
    import torch
    from datasets import Dataset
    from peft import LoraConfig as PeftLoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from trl import SFTConfig, SFTTrainer

    bnb = BitsAndBytesConfig(
        load_in_4bit=TRAIN.load_in_4bit,
        bnb_4bit_quant_type=TRAIN.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained(TRAIN.base_model)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        TRAIN.base_model, quantization_config=bnb, device_map="auto",
    )
    model = get_peft_model(model, PeftLoraConfig(
        r=LORA.r, lora_alpha=LORA.lora_alpha, lora_dropout=LORA.lora_dropout,
        target_modules=list(LORA.target_modules), task_type="CAUSAL_LM",
    ))

    train_ds = Dataset.from_list([{"text": build_prompt(e)} for e in load_split("train")])
    val_ds = Dataset.from_list([{"text": build_prompt(e)} for e in load_split("val")])

    out_dir = ADAPTERS_DIR / task
    cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=TRAIN.epochs,
        learning_rate=TRAIN.lr,
        lr_scheduler_type=TRAIN.lr_scheduler,
        per_device_train_batch_size=TRAIN.per_device_train_batch_size,
        gradient_accumulation_steps=TRAIN.gradient_accumulation_steps,
        save_steps=TRAIN.save_steps,
        logging_steps=10,
        eval_strategy="steps",
        bf16=True,
        seed=SEED,
        report_to="wandb",
        run_name=f"{TRAIN.wandb_project}-{task}",
    )
    trainer = SFTTrainer(model=model, args=cfg,
                         train_dataset=train_ds, eval_dataset=val_ds)
    trainer.train()           # resume_from_checkpoint=True on Colab restart
    trainer.save_model(str(out_dir))
    _register(task, out_dir)
    return out_dir


def _register(task: str, out_dir: Path) -> None:
    """Add/update adapter registry entry (metrics filled by eval)."""
    import datetime as dt
    reg_path = ADAPTERS_DIR / "registry.json"
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else {}
    reg[task] = {
        "adapter_path": str(out_dir.relative_to(ADAPTERS_DIR.parent)),
        "base_model": TRAIN.base_model,
        "metrics": {},
        "created_at": dt.datetime.utcnow().isoformat() + "Z",
    }
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    reg_path.write_text(json.dumps(reg, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="financial_extraction")
    args = ap.parse_args()
    out = train(args.task)
    print(f"[ok] adapter saved -> {out}")


if __name__ == "__main__":
    main()
