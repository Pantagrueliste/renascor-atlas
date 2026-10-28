#!/bin/bash
# Open instruction LLM, zero-shot, on the 300-passage sample (local GPU, offline). ARGS in the environment: --name --model --sample.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cag30/renascor-atlas-data/ner/logs/%x-%j.out
set -uo pipefail
set -f
D="$HOME/renascor-atlas-data/ner"
source "$D/${VENV:-venv_train}/bin/activate"
export PYTHONPATH="$HOME/renascor-atlas/src" HF_HUB_CACHE="$D/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
echo "[ner-llm] $(date +%T) $(hostname) gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
nice -n 5 python -m ratlas.ner.llm run $ARGS --out "$D/runs"
rc=$?
echo "[ner-llm] exit $rc $(date +%T)"
exit $rc
