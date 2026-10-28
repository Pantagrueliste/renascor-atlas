#!/bin/bash
# v2.2 trial of cards_globe (five-way kinds, letters.json) on the trial tables of v22_trial.sbatch and the v2.1 mentions; written to scratch_v22/cards.
#SBATCH --job-name=ratlas-v22-trial-cards
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:00:00
set -euo pipefail
D="$HOME/renascor-atlas-data"; S="$D/scratch_v22"; C="$S/cards"
mkdir -p "$C/in" "$C/out"
for f in globe_places.json places_global.parquet places_local_geo.parquet; do cp "$S/final/$f" "$C/in/"; done
for f in meta_w320.json wd_class_labels.tsv wd_entities.parquet wd_images.tsv wd_places.parquet wd_places_issues.tsv; do cp "$D/cards_v21/in/$f" "$C/in/"; done
cd "$HOME/renascor-atlas"
nice -n 10 .venv/bin/python -I -m ratlas.cards_globe build \
  --final "$C/in" --mentions "$D/out_v2_1/all/mentions.parquet" --documents "$D/docs_v21/cards/documents.parquet" \
  --cards "$C/in" --entries "$D/docs_cards/entries" --tgn "$D/tgn" --gn-index "$D/geocode_v2_1/gn_index.parquet" \
  --wd-entities "$C/in/wd_entities.parquet" --wd-class-labels "$C/in/wd_class_labels.tsv" \
  --images "$C/in/wd_images.tsv" --img-dir "$D/cards/img" --img-meta "$C/in/meta_w320.json" \
  --issues "$C/in/wd_places_issues.tsv" --out "$C/out"
