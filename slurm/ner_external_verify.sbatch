#!/bin/bash
# Integrity + config/tokenizer check of the pre-downloaded models (no weights loaded, no torch). CPU only.
#SBATCH --job-name=ratlas-ner-ext-verify
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:50:00
#SBATCH --output=/home/cag30/renascor-atlas-data/ner/logs/%x-%j.out
set -euo pipefail
D="$HOME/renascor-atlas-data/ner"
cd "$HOME/renascor-atlas"
echo "[ner ext verify] $(date +%T) on $(hostname)"
nice -n 10 "$D/venv_eval/bin/python" -I scripts/ner_external_verify_models.py --cache "$D/hf_cache"
echo "[ner ext verify] done $(date +%T)"
