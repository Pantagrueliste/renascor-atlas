#!/bin/bash
# One NER fine-tuning run on one GPU. The arguments of ratlas.ner.train are in the environment variable ARGS (set by scripts/ner_submit_train.sh):
#   sbatch --partition=gpu.L40S --job-name=ner-xlmr_base --export=ALL,ARGS="--name xlmr_base --model ..." slurm/ner_train.sbatch
# Offline (the models are in hf_cache), bf16, checkpointed: when the time limit is near the run saves its state and exits with 3; resubmit the same command to resume.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cag30/renascor-atlas-data/ner/logs/%x-%j.out
set -uo pipefail
set -f   # ARGS holds glob patterns for the Python side: the shell must not expand them
D="$HOME/renascor-atlas-data/ner"
source "$D/venv_train/bin/activate"
export PYTHONPATH="$HOME/renascor-atlas/src" HF_HUB_CACHE="$D/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
mkdir -p "$D/runs" "$D/models" "$D/cache" "$D/logs"
echo "[ner-train] $(date +%T) $(hostname) job ${SLURM_JOB_ID:-} gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[ner-train] ARGS=$ARGS"
nice -n 5 python -m ratlas.ner.train $ARGS --out "$D/runs" --models "$D/models" --cache "$D/cache" --time-limit-min "${LIMIT_MIN:-215}"
rc=$?
echo "[ner-train] exit $rc $(date +%T)"
exit $rc
