#!/bin/bash
# Reference set v2 (gold2): Qwen3.5-9B zero-shot on all passages (needs the newer transformers of venv_llm; submit with VENV=venv_llm).
set -u
mkdir -p "$OUT"
python scripts/ner2_predict_llm.py --model Qwen/Qwen3.5-9B --data "$PASSAGES" --out "$OUT/llm_qwen35_9b.jsonl.gz" --answers "$OUT/llm_qwen35_9b_answers.jsonl"
