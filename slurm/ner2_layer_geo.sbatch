#!/bin/bash
# NER layer v1, geocoding of the recommended spans (CPU only, debug partition). Steps: prep | index | geocode | points
#   sbatch --output=$HOME/renascor-atlas-data/logs/%x-%j.out slurm/ner2_layer_geo.sbatch STEP [args]
#SBATCH --job-name=ner_geo
#SBATCH --partition=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
set -uo pipefail
STEP="${1:?step}"; shift || true
D="$HOME/renascor-atlas-data"; L="$D/ner/layer_v1"; GEO="$L/geo"
cd "$HOME/renascor-atlas"
PY="nice -n 10 .venv/bin/python -I"
echo "[ner_geo $STEP] $(date +%T) on $(hostname)"
case "$STEP" in
  prep)    $PY scripts/ner2_layer_geo_prep.py --layer "$L" --out "$GEO" "$@" ;;
  index)   $PY scripts/ner2_layer_geocode.py --forms "$GEO/forms.parquet" --doc-forms "$GEO/doc_forms.parquet" --out "$GEO/gc" --index-only "$@" ;;
  geocode) $PY scripts/ner2_layer_geocode.py --forms "$GEO/forms.parquet" --doc-forms "$GEO/doc_forms.parquet" --out "$GEO/gc" "$@" ;;
  points)  $PY scripts/ner2_layer_geo_points.py --layer "$L" --geo "$GEO" "$@" ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
rc=$?; echo "[ner_geo $STEP] done rc=$rc $(date +%T)"; exit $rc
