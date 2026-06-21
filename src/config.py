"""Central config. Env-overridable, no hardcoded secrets."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ADAPTERS_DIR = ROOT / "adapters"

SEED = 42

# valid sentiment labels — the output contract
SENTIMENTS = ("positive", "negative", "neutral")


@dataclass(frozen=True)
class DataConfig:
    total_target: int = 1200          # 800–1500 band
    splits: tuple[float, float, float] = (0.8, 0.1, 0.1)  # train/val/test
    stratify_by: str = "sentiment"
    hf_sources: tuple[str, ...] = (
        "financial_phrasebank",
        "zeroshot/twitter-financial-news-sentiment",
    )
    local_labeled: str = "indian_market.jsonl"   # the differentiator set


@dataclass(frozen=True)
class LoraConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass(frozen=True)
class TrainConfig:
    base_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    epochs: int = 3
    lr: float = 2e-4
    lr_scheduler: str = "cosine"
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    save_steps: int = 50              # checkpoint cadence (Colab-timeout guard)
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    wandb_project: str = "llm-finetune-fin-extract"


@dataclass(frozen=True)
class Secrets:
    hf_token: str = field(default_factory=lambda: os.getenv("HF_TOKEN", ""))
    wandb_key: str = field(default_factory=lambda: os.getenv("WANDB_API_KEY", ""))
    anthropic_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))


DATA = DataConfig()
LORA = LoraConfig()
TRAIN = TrainConfig()
