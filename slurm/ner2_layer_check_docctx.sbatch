#!/bin/bash
# NER layer v1, independent check, step 3 (CPU only, debug partition).
#SBATCH --job-name=chk_layer_docctx
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -uo pipefail
L="$HOME/renascor-atlas-data/ner/layer_v1"
cd "$HOME/renascor-atlas"
nice -n 10 .venv/bin/python -I scripts/ner2_layer_check_docctx.py --layer "$L" --items "$L/check/judged_items.json" --out "$L/check/docctx.json"
