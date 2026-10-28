#!/bin/bash
# Document table (cards): keys / build stages on a compute node. Usage (on Hypatia):
#   sbatch --output=$HOME/renascor-atlas-data/logs/%x-%j.out slurm/docs_table.sbatch keys --mentions ... --out ...
#SBATCH --job-name=ratlas-docs
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
set -euo pipefail
export RATLAS_DATA="$HOME/renascor-atlas-data"
export RATLAS_RAW="$RATLAS_DATA/raw"
export RATLAS_SCAN="$RATLAS_DATA/scan.jsonl.gz"
cd "$HOME/renascor-atlas"
nice -n 10 .venv/bin/python -I -m ratlas.docs_table "$@"
