#!/bin/bash
# One array task per slug. Do not sbatch this directly: use slurm/submit.sh slug1 slug2 ...
# Defaults: debug partition, 8 cpus, 16G, 30 min. Override on the sbatch command line (see submit.sh).
#SBATCH --job-name=ratlas
#SBATCH --partition=debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
# (the log path is set by submit.sh: ~/renascor-atlas-data/logs/%x-%A_%a.out)
set -euo pipefail
: "${SLUGS:?SLUGS (space-separated) must be exported, use slurm/submit.sh}"
read -ra ARR <<< "$SLUGS"
SLUG="${ARR[$SLURM_ARRAY_TASK_ID]}"
export RATLAS_DATA="$HOME/renascor-atlas-data"
export RATLAS_RAW="$RATLAS_DATA/raw"
export RATLAS_SCAN="$RATLAS_DATA/scan.jsonl.gz"
export RATLAS_SCAN_EXTRA="$RATLAS_DATA/scan_extra.jsonl.gz"     # re-harvest of 2026-10-17 (v2)
export RATLAS_OUT="$RATLAS_DATA/${OUTNAME:-out}"      # OUTNAME=out_v2 for the v2 rebuild
cd "$HOME/renascor-atlas"
echo "ratlas extract $SLUG on $(hostname) with ${SLURM_CPUS_PER_TASK:-1} cpus, limit=${LIMIT:-none}"
nice -n 10 .venv/bin/python -I -m ratlas extract "$SLUG" --workers "${SLURM_CPUS_PER_TASK:-1}" ${LIMIT:+--limit "$LIMIT"}
