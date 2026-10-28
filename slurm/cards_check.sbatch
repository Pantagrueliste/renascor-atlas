#!/bin/bash
# Card table checks (read-only) on a compute node. Usage (on Hypatia):
#   sbatch --output=$HOME/renascor-atlas-data/logs/%x-%j.out slurm/cards_check.sbatch mentions --mentions ... --documents ...
#SBATCH --job-name=ratlas-cardcheck
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
set -uo pipefail
export RATLAS_DATA="$HOME/renascor-atlas-data"
cd "$HOME/renascor-atlas"
nice -n 10 .venv/bin/python -I -m ratlas.cards_check "$@"
