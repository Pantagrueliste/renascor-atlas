#!/bin/bash
# Consistency check of the converted external NER datasets. CPU only.
#SBATCH --job-name=ratlas-ner-ext-check
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cag30/renascor-atlas-data/ner/logs/%x-%j.out
set -euo pipefail
cd "$HOME/renascor-atlas"
nice -n 10 .venv/bin/python -I scripts/ner_external_check.py --root "$HOME/renascor-atlas-data/ner/external"
