#!/bin/bash
# Aggregate the converted external NER datasets into training views (hist_gold, hist_open, modern_silver, all_private). CPU only.
#SBATCH --job-name=ratlas-ner-ext-aggregate
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/home/cag30/renascor-atlas-data/ner/logs/%x-%j.out
set -euo pipefail
D="$HOME/renascor-atlas-data/ner"
cd "$HOME/renascor-atlas"
echo "[ner ext aggregate] $(date +%T) on $(hostname)"
nice -n 10 .venv/bin/python -I scripts/ner_external_aggregate.py --root "$D/external"
echo "[ner ext aggregate] done $(date +%T)"
