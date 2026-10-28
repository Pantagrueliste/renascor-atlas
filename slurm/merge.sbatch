#!/bin/bash
# Integration merge on a compute node: sbatch --output=$HOME/renascor-atlas-data/logs/%x-%j.out slurm/merge.sbatch
#SBATCH --job-name=ratlas-merge
#SBATCH --partition=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:30:00
set -euo pipefail
export RATLAS_DATA="$HOME/renascor-atlas-data"
export RATLAS_RAW="$RATLAS_DATA/raw"
export RATLAS_SCAN="$RATLAS_DATA/scan.jsonl.gz"
export RATLAS_OUT="$RATLAS_DATA/out"
cd "$HOME/renascor-atlas"
nice -n 10 .venv/bin/python -I -m ratlas merge "$@"
