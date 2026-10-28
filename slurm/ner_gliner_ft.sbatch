#!/bin/bash
# GLiNER fine-tuning (ratlas.ner.gliner_ft), one GPU. ARGS in the environment (set by scripts/ner_submit_gliner_ft.sh).
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cag30/renascor-atlas-data/ner/logs/%x-%j.out
set -uo pipefail
set -f
D="$HOME/renascor-atlas-data/ner"
source "$D/venv_train/bin/activate"
export PYTHONPATH="$HOME/renascor-atlas/src" HF_HUB_CACHE="$D/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
echo "[ner-gliner-ft] $(date +%T) $(hostname) job ${SLURM_JOB_ID:-} gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[ner-gliner-ft] ARGS=$ARGS"
nice -n 5 python -m ratlas.ner.gliner_ft $ARGS --out "$D/runs" --models "$D/models"
rc=$?
echo "[ner-gliner-ft] exit $rc $(date +%T)"
exit $rc
