#!/bin/bash
# Public NER datasets (official sources only) -> ~/renascor-atlas-data/ner/external/<name>/raw + SOURCES.tsv. Network only, 1 cpu.
#   sbatch slurm/ner_external_fetch.sbatch                                    all sources except the 15 GB medieval_charters
#   sbatch --export=ALL,ONLY=medieval_charters slurm/ner_external_fetch.sbatch
#SBATCH --job-name=ratlas-ner-ext-fetch
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:50:00
#SBATCH --output=/home/cag30/renascor-atlas-data/ner/logs/%x-%j.out
set -euo pipefail
D="$HOME/renascor-atlas-data/ner"
cd "$HOME/renascor-atlas"
if [ -z "${ONLY:-}" ]; then
  ONLY=$("$D/venv_hf/bin/python" -I scripts/ner_external_fetch.py --list | cut -f1 | grep -v '^medieval_charters$' | paste -sd, -)
fi
echo "[ner ext fetch] $(date +%T) on $(hostname): $ONLY"
nice -n 10 "$D/venv_hf/bin/python" -I scripts/ner_external_fetch.py --only "$ONLY"
echo "[ner ext fetch] done $(date +%T)"
