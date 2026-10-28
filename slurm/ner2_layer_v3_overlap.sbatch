#!/bin/bash
#SBATCH --job-name=ner_v3_overlap
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
cd "$HOME/renascor-atlas"
D="$HOME/renascor-atlas-data/ner"
nice -n 10 .venv/bin/python -I scripts/ner2_layer_v3_overlap_check.py --v3 "$D/layer_v3" --data "$D" --out "$D/layer_v3/dedup_all/overlap_check.json"
