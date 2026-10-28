#!/bin/bash
# Convert the fetched public NER datasets to our JSONL format (LOC / PER). CPU only.
#   sbatch slurm/ner_external_convert.sbatch                         all datasets
#   sbatch --export=ALL,ONLY=hipe2022,kind slurm/ner_external_convert.sbatch
#SBATCH --job-name=ratlas-ner-ext-convert
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:50:00
#SBATCH --output=/home/cag30/renascor-atlas-data/ner/logs/%x-%j.out
set -euo pipefail
D="$HOME/renascor-atlas-data/ner"
cd "$HOME/renascor-atlas"
echo "[ner ext convert] $(date +%T) on $(hostname) only=${ONLY:-all}"
nice -n 10 .venv/bin/python -I scripts/ner_external_convert.py --root "$D/external" ${ONLY:+--only "$ONLY"}
echo "[ner ext convert] done $(date +%T)"
