#!/bin/bash
# Data shards of the private globe page (cards_globe.py). On Hypatia: sbatch --output=$HOME/renascor-atlas-data/logs/%x-%j.out slurm/cards_globe.sbatch
# Inputs in $CARDS/in (final tables of the atlas, card inputs), documents in $DOCS/cards, output in $CARDS/out.
#SBATCH --job-name=ratlas-globe
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:00:00
set -euo pipefail
D="$HOME/renascor-atlas-data"
CARDS="${CARDS:-$D/cards_v21}"
DOCS="${DOCS:-$D/docs_v21}"
cd "$HOME/renascor-atlas"
nice -n 10 .venv/bin/python -I -m ratlas.cards_globe build \
  --final "$CARDS/in" --mentions "$D/out_v2_1/all/mentions.parquet" --documents "$DOCS/cards/documents.parquet" \
  --cards "$CARDS/in" --entries "$D/docs_cards/entries" --tgn "$D/tgn" --gn-index "$D/geocode_v2_1/gn_index.parquet" \
  --wd-entities "$CARDS/in/wd_entities.parquet" --wd-class-labels "$CARDS/in/wd_class_labels.tsv" \
  --images "$CARDS/in/wd_images.tsv" --img-dir "$D/cards/img" --img-meta "$CARDS/in/meta_w320.json" \
  --issues "$CARDS/in/wd_places_issues.tsv" --out "$CARDS/out"
