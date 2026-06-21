"""Shared model loading + generation. Used by serve.py and evaluate.py.

One base model in 4-bit, multiple LoRA adapters hot-swapped via set_adapter().
Heavy deps imported lazily so the module stays importable without a GPU stack.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .config import ADAPTERS_DIR, TRAIN
from .schema import INSTRUCTION, parse_and_validate

GEN_KWARGS = dict(max_new_tokens=256, do_sample=False, temperature=0.0)


def build_prompt(text: str) -> str:
    """Inference-time prompt — matches train.py format minus the target."""
    return f"<|instruction|>\n{INSTRUCTION}\n<|input|>\n{text}\n<|output|>\n"


def load_registry() -> dict:
    path = ADAPTERS_DIR / "registry.json"
    return json.loads(path.read_text()) if path.exists() else {}


class Engine:
    """Base model + adapters loaded once. Swap adapter per call."""

    def __init__(self, load_4bit: bool = True):
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  BitsAndBytesConfig)

        quant = None
        if load_4bit:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=TRAIN.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        self.tok = AutoTokenizer.from_pretrained(TRAIN.base_model)
        self.tok.pad_token = self.tok.pad_token or self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            TRAIN.base_model, quantization_config=quant, device_map="auto",
        )
        self._adapters: set[str] = set()
        for task, meta in load_registry().items():
            self.attach(task, Path(meta["adapter_path"]))

    def attach(self, task: str, adapter_path: Path) -> None:
        """Load a LoRA adapter under a name (idempotent)."""
        from peft import PeftModel
        if task in self._adapters:
            return
        if not self._adapters:
            self.model = PeftModel.from_pretrained(
                self.model, str(adapter_path), adapter_name=task)
        else:
            self.model.load_adapter(str(adapter_path), adapter_name=task)
        self._adapters.add(task)

    def has(self, task: str) -> bool:
        return task in self._adapters

    def generate(self, text: str, task: str | None = None, retry: bool = True) -> str:
        """Run inference. task=None -> base model (zero-shot baseline)."""
        if task and self.has(task):
            self.model.set_adapter(task)          # hot-swap, no base reload
        elif task:
            raise KeyError(f"adapter not loaded: {task}")

        out = self._decode(build_prompt(text))
        if retry and not parse_and_validate(out).ok:
            out = self._decode(build_prompt(text) + "{")   # nudge into JSON
            out = out if out.lstrip().startswith("{") else "{" + out
        return out

    def _decode(self, prompt: str) -> str:
        import torch
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            ids = self.model.generate(**inputs, **GEN_KWARGS,
                                      pad_token_id=self.tok.pad_token_id)
        gen = ids[0][inputs["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True).strip()


@lru_cache(maxsize=1)
def get_engine(load_4bit: bool = True) -> Engine:
    return Engine(load_4bit=load_4bit)
