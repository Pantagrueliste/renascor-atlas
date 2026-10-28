#!/bin/bash
# Independent audit, one GPU job: JOB_SH (a bash script path, relative to ~/renascor-atlas) runs inside the training environment (offline, models in hf_cache).
#   sbatch --partition=gpu.L40S --job-name=ner-audit-x --export=ALL,JOB_SH=scripts/ner_audit_run_top.sh slurm/ner_audit_gpu.sbatch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/cag30/renascor-atlas-data/ner/logs/%x-%j.out
set -uo pipefail
D="$HOME/renascor-atlas-data/ner"
source "$D/${VENV:-venv_train}/bin/activate"
export PYTHONPATH="$HOME/renascor-atlas/src" HF_HUB_CACHE="$D/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 ND="$D"
cd "$HOME/renascor-atlas"
echo "[ner-audit-gpu] $(date +%T) $(hostname) job ${SLURM_JOB_ID:-} gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) JOB_SH=$JOB_SH"
nice -n 5 bash "$JOB_SH"
rc=$?
echo "[ner-audit-gpu] exit $rc $(date +%T)"
exit $rc
