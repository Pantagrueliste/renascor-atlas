#!/bin/bash
# Data shards of the private globe page (cards_globe.py). On Hypatia: sbatch --output=$HOME/renascor-atlas-data/logs/%x-%j.out slurm/cards_globe_v23.sbatch
# v2.3: inputs are staged into $CARDS/in (final tables of out_v2_3, card inputs of cards_v21), documents in $DOCS/cards (docs_v21: same keys), output in $CARDS/out
# (core.json, wd.json, docs/, img/, letters.json, letters_coverage.tsv, build_report.json).
#SBATCH --job-name=ratlas-globe
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:00:00
set -euo pipefail
D="$HOME/renascor-atlas-data"
CARDS="${CARDS:-$D/cards_v23}"
DOCS="${DOCS:-$D/docs_v21}"
# inputs: the final tables of v2.3 and the card inputs of v2.1 (Wikidata cards, images, class labels: fetched on 2026-10-16/17, no network now)
mkdir -p "$CARDS/in" "$CARDS/out"
for f in globe_places.json places_global.parquet places_local_geo.parquet; do cp "$D/out_v2_3/reconcile/final/$f" "$CARDS/in/"; done
for f in meta_w320.json wd_class_labels.tsv wd_entities.parquet wd_images.tsv wd_places.parquet wd_places_issues.tsv; do cp "$D/cards_v21/in/$f" "$CARDS/in/"; done
cd "$HOME/renascor-atlas"
nice -n 10 .venv/bin/python -I -m ratlas.cards_globe build \
  --final "$CARDS/in" --mentions "$D/out_v2_3/all/mentions.parquet" --documents "$DOCS/cards/documents.parquet" \
  --cards "$CARDS/in" --entries "$D/docs_cards/entries" --tgn "$D/tgn" --gn-index "$D/geocode_v2_3/gn_index.parquet" \
  --wd-entities "$CARDS/in/wd_entities.parquet" --wd-class-labels "$CARDS/in/wd_class_labels.tsv" \
  --images "$CARDS/in/wd_images.tsv" --img-dir "$D/cards/img" --img-meta "$CARDS/in/meta_w320.json" \
  --issues "$CARDS/in/wd_places_issues.tsv" --out "$CARDS/out"
