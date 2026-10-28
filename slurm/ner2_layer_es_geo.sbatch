#!/bin/bash
# Spanish rerun (mDeBERTa main): merge | prep | index | geocode | points, same geocoder scripts as the layer, all under layer_v1/es_mdeberta/ (CPU, debug partition).
#   GEO_SUB=geo_strict MINCONF=0.98 sbatch --export=ALL ... for the strict variant (default: geo, 0.5)
#   sbatch --output=$HOME/renascor-atlas-data/ner/layer_v1/es_mdeberta/logs/%x-%j.out slurm/ner2_layer_es_geo.sbatch STEP [args]
#SBATCH --job-name=ner_es_geo
#SBATCH --partition=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
set -uo pipefail
STEP="${1:?step}"; shift || true
D="$HOME/renascor-atlas-data"; L="$D/ner/layer_v1"; E="$L/es_mdeberta"; GEO="$E/${GEO_SUB:-geo}"; MINCONF="${MINCONF:-0.5}"
cd "$HOME/renascor-atlas"
PY="nice -n 10 .venv/bin/python -I"
echo "[ner_es_geo $STEP] $(date +%T) on $(hostname)"
case "$STEP" in
  merge)   $PY scripts/ner2_layer_es_merge.py --layer "$L" --es "$E" "$@" ;;
  prep)    $PY scripts/ner2_layer_geo_prep.py --layer "$E" --out "$GEO" --min-conf "$MINCONF" --min-agree 2 "$@" ;;
  index)   $PY scripts/ner2_layer_geocode.py --forms "$GEO/forms.parquet" --doc-forms "$GEO/doc_forms.parquet" --out "$GEO/gc" --index-only "$@" ;;
  geocode) $PY scripts/ner2_layer_geocode.py --forms "$GEO/forms.parquet" --doc-forms "$GEO/doc_forms.parquet" --out "$GEO/gc" "$@" ;;
  points)  $PY scripts/ner2_layer_geo_points.py --layer "$E" --geo "$GEO" "$@" ;;
  all)     for s in merge prep index geocode points; do bash "$0" $s || exit $?; done ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
rc=$?; echo "[ner_es_geo $STEP] done rc=$rc $(date +%T)"; exit $rc
