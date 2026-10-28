#!/bin/bash
# Usage: sbatch -J teacher-<name> [--gres=gpu:2 for tp 2] scripts/ner3_teacher_job.sh <model> <prompt zero|guided> <name> [tp]
# GPU job (partition per our rules: gpu.A100, <=4h, at most 2 of ours at once). Needs the venv with vllm (to identify first) or --engine hf with venv_llm.
#SBATCH -p gpu.A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 04:00:00
set -euo pipefail
MODEL=$1; PROMPT=$2; NAME=$3; TP=${4:-1}
cd ~/renascor-atlas
source ~/renascor-atlas-data/venv_llm/bin/activate 2>/dev/null || source ~/venv_llm/bin/activate
python scripts/ner3_teacher_run.py --model "$MODEL" --prompt "$PROMPT" --tp "$TP" --data ~/renascor-atlas-data/ner/v3/dev3/passages.jsonl --out ~/renascor-atlas-data/ner/v3/teacher/$NAME
