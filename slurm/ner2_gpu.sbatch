#!/bin/bash
# Reference set v2: one GPU job running JOB_SH (path relative to ~/renascor-atlas) inside the NER environment (offline, models in hf_cache).
#   sbatch --partition=gpu.L40S --job-name=ner2_gold2_pred --export=ALL,JOB_SH=scripts/ner2_run_predict.sh,PASSAGES=...,OUT=... slurm/ner2_gpu.sbatch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cag30/renascor-atlas-data/ner/logs/%x-%j.out
set -uo pipefail
D="$HOME/renascor-atlas-data/ner"
source "$D/${VENV:-venv_train}/bin/activate"
export PYTHONPATH="$HOME/renascor-atlas/src" HF_HUB_CACHE="$D/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 ND="$D"
cd "$HOME/renascor-atlas"
echo "[ner2-gpu] $(date +%T) $(hostname) job ${SLURM_JOB_ID:-} gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) JOB_SH=$JOB_SH"
J="$(mktemp /tmp/ner2job.XXXXXX.sh)"; cp "$JOB_SH" "$J"      # a copy: the script on disk may be edited while the job runs
nice -n 5 bash "$J"
rc=$?
echo "[ner2-gpu] exit $rc $(date +%T)"
exit $rc
