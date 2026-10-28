#!/bin/bash
# Data shards of the private globe page (cards_globe.py). On Hypatia: sbatch --output=$HOME/renascor-atlas-data/logs/%x-%j.out slurm/cards_globe_v24.sbatch
# v2.4: inputs are staged into $CARDS/in (final tables of out_v2_4, card inputs of cards_v21), documents in $DOCS/cards (docs_v21: same keys), output in $CARDS/out
# (core.json, wd.json, docs/, img/, letters.json, letters_coverage.tsv, build_report.json).
#SBATCH --job-name=ratlas-globe
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -euo pipefail
D="$HOME/renascor-atlas-data"
CARDS="${CARDS:-$D/cards_v24}"
DOCS="${DOCS:-$D/docs_v21}"
# inputs: the final tables of v2.4 and the card inputs of v2.1 (Wikidata cards, images, class labels: fetched on 2026-10-16/17, no network now)
mkdir -p "$CARDS/in" "$CARDS/out"
for f in globe_places.json places_global.parquet places_local_geo.parquet; do cp "$D/out_v2_4/reconcile/final/$f" "$CARDS/in/"; done
for f in meta_w320.json wd_class_labels.tsv wd_entities.parquet wd_images.tsv wd_places.parquet wd_places_issues.tsv; do cp "$D/cards_v21/in/$f" "$CARDS/in/"; done
cd "$HOME/renascor-atlas"
nice -n 10 .venv/bin/python -I -m ratlas.cards_globe build \
  --final "$CARDS/in" --mentions "$D/out_v2_4/all/mentions.parquet" --documents "$DOCS/cards/documents.parquet" \
  --cards "$CARDS/in" --entries "$D/docs_cards/entries" --tgn "$D/tgn" --gn-index "$D/geocode_v2_4/gn_index.parquet" \
  --wd-entities "$CARDS/in/wd_entities.parquet" --wd-class-labels "$CARDS/in/wd_class_labels.tsv" \
  --images "$CARDS/in/wd_images.tsv" --img-dir "$D/cards/img" --img-meta "$CARDS/in/meta_w320.json" \
  --issues "$CARDS/in/wd_places_issues.tsv" --out "$CARDS/out"
# v2.4: extra.json next to core.json (fine place type, top 10 variants with language / edition / count, mentions located by a name per row; docs/NORMALISATION.md) and its tables
nice -n 10 .venv/bin/python -I -m ratlas.name_variants extra \
  --final "$CARDS/in" --mentions "$D/out_v2_4/all/mentions.parquet" --core "$CARDS/out/core.json" \
  --wd-entities "$CARDS/in/wd_entities.parquet" --gn-index "$D/geocode_v2_4/gn_index.parquet" \
  --out "$CARDS/out/extra.json" --globals-out "$CARDS/out/global_places_names_types.tsv"
# every source term with 20 or more placed mentions needs a reviewed row of data/placetypes.tsv (exit 2 and a list otherwise: the job is not failed by it, the report says so)
nice -n 10 .venv/bin/python -I -m ratlas.placetypes inventory --final "$CARDS/in" --wd-entities "$CARDS/in/wd_entities.parquet" --gn-index "$D/geocode_v2_4/gn_index.parquet" \
  --out "$CARDS/out/placetype_coverage.tsv" || echo "placetypes inventory exit code $?"
