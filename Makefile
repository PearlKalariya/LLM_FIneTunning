.PHONY: install data train eval serve test

install:
	pip install -r requirements.txt

data:          ## curate -> stratified split -> data/processed/*.jsonl
	python -m src.data_prep

train:         ## QLoRA fine-tune (run on T4)
	python -m src.train --task financial_extraction

eval:          ## baseline vs fine-tuned -> eval_results.json
	python -m src.evaluate --task financial_extraction

serve:         ## FastAPI on :8020
	uvicorn src.serve:app --host 0.0.0.0 --port 8020

test:
	pytest -q
