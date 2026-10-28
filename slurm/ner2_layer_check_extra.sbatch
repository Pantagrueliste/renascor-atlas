#!/bin/bash
# NER layer v1, independent check (CPU only, debug partition): second draw and suspect forms.
#   sbatch --output=$HOME/renascor-atlas-data/logs/%x-%j.out slurm/ner2_layer_check.sbatch
#SBATCH --job-name=chk_layer_extra
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=01:30:00
set -uo pipefail
L="$HOME/renascor-atlas-data/ner/layer_v1"
cd "$HOME/renascor-atlas"
nice -n 10 .venv/bin/python -I scripts/ner2_layer_check_extra.py --layer "$L" --out "$L/check"
