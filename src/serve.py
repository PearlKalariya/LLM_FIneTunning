"""FastAPI multi-adapter serving (Week 4).

Run:  uvicorn src.serve:app --port 8020
Base model loaded once; LoRA adapters hot-swapped per request via set_adapter().
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import TRAIN
from .inference import load_registry
from .schema import parse_and_validate

MAX_INPUT_CHARS = 4000  # input size cap (see SECURITY.md)
SERVE_4BIT = os.getenv("SERVE_4BIT", "1") != "0"

app = FastAPI(title="LLM Fine-Tune Platform", version="0.1.0")

_state: dict = {"engine": None, "registry": {}}


class InferRequest(BaseModel):
    task: str = Field(..., examples=["financial_extraction"])
    text: str = Field(..., min_length=1, max_length=MAX_INPUT_CHARS)


@app.on_event("startup")
def _startup() -> None:
    _state["registry"] = load_registry()
    if not _state["registry"]:
        print("[warn] empty adapter registry — train first; /infer will 404")
        return
    try:
        from .inference import get_engine
        _state["engine"] = get_engine(load_4bit=SERVE_4BIT)  # base + adapters once
    except Exception as e:  # serving stays up for /health even if GPU stack absent
        print(f"[warn] engine not loaded: {e}")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "base_model": TRAIN.base_model,
        "model_loaded": _state["engine"] is not None,
        "adapters": list(_state["registry"].keys()),
    }


@app.get("/adapters")
def adapters() -> dict:
    return _state["registry"]


@app.post("/infer")
def infer(req: InferRequest) -> dict:
    if req.task not in _state["registry"]:
        raise HTTPException(404, f"unknown task: {req.task}")
    eng = _state["engine"]
    if eng is None:
        raise HTTPException(503, "model not loaded")

    out = eng.generate(req.text, task=req.task, retry=True)  # one constrained retry
    res = parse_and_validate(out)
    if not res.ok:
        raise HTTPException(422, f"model output failed schema: {res.error}")
    return res.data
