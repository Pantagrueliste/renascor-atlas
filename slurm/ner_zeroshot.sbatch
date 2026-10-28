#!/bin/bash
# One off-the-shelf model (no training) on one GPU. Environment: BACKEND (hf | gliner | flair), VENV (venv_train | venv_flair | venv_gl2), ARGS (ratlas.ner.zeroshot arguments).
#   sbatch --partition=gpu.A30 --job-name=ner-zs-x --export=ALL,BACKEND=hf,VENV=venv_train,ARGS="--name zs_x --model ..." slurm/ner_zeroshot.sbatch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cag30/renascor-atlas-data/ner/logs/%x-%j.out
set -uo pipefail
set -f   # ARGS holds glob patterns for the Python side: the shell must not expand them
D="$HOME/renascor-atlas-data/ner"
source "$D/${VENV:-venv_train}/bin/activate"
export PYTHONPATH="$HOME/renascor-atlas/src" HF_HUB_CACHE="$D/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
mkdir -p "$D/runs" "$D/logs"
echo "[ner-zs] $(date +%T) $(hostname) job ${SLURM_JOB_ID:-} backend $BACKEND"
echo "[ner-zs] ARGS=$ARGS"
nice -n 5 python -m ratlas.ner.zeroshot --backend "$BACKEND" $ARGS --out "$D/runs"
rc=$?
echo "[ner-zs] exit $rc $(date +%T)"
exit $rc
