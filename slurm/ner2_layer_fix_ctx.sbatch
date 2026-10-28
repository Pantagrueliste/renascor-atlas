#!/bin/bash
# NER layer v2 fix, step 1 (CPU only, debug partition): mean person probability per form, context cues, titles of all documents.
#   sbatch --output=$HOME/renascor-atlas-data/logs/%x-%j.out slurm/ner2_layer_fix_ctx.sbatch [--peek 15]
#SBATCH --job-name=ner_fix_ctx
#SBATCH --partition=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
set -uo pipefail
L="$HOME/renascor-atlas-data/ner/layer_v1"
cd "$HOME/renascor-atlas"
echo "[ner_fix_ctx] $(date +%T) on $(hostname)"
nice -n 10 .venv/bin/python -I scripts/ner2_layer_fix_ctx.py --layer "$L" --out "$L/fix" "$@"
rc=$?; echo "[ner_fix_ctx] done rc=$rc $(date +%T)"; exit $rc
