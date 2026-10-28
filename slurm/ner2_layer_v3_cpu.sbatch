#!/bin/bash
# NER layer v3 (TEI editions + repaired extra documents), CPU steps (debug partition): merge | prep | ctx | index | geocode | points | tables | all
# Everything lands under ~/renascor-atlas-data/ner/layer_v3/ (tei/, extra/, geo/). The geocoder scripts are those of the layer v1; the geocode step gets the edition / document
# context (ctx_map, doc_link) of ner2_layer_v3_geo_ctx.py.
#   sbatch --output=$HOME/renascor-atlas-data/ner/layer_v3/logs/%x-%j.out slurm/ner2_layer_v3_cpu.sbatch STEP [args]
#SBATCH --job-name=ner_v3_cpu
#SBATCH --partition=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
set -uo pipefail
STEP="${1:?step}"; shift || true
D="$HOME/renascor-atlas-data"; V3="$D/ner/layer_v3"; GEO="$V3/${GEO_NAME:-geo}"; LAY="$GEO/layer"
cd "$HOME/renascor-atlas"
PY="nice -n 10 .venv/bin/python -I"
echo "[ner_v3 $STEP] $(date +%T) on $(hostname)"
run() {
case "$1" in
  merge)   shift; $PY scripts/ner2_layer_v3_merge.py --v3 "$V3" --geo-name "${GEO_NAME:-geo}" "$@" ;;
  prep)    shift; $PY scripts/ner2_layer_geo_prep.py --layer "$LAY" --out "$GEO" --min-conf 0.0 --min-agree 2 "$@" ;;
  ctx)     shift; $PY scripts/ner2_layer_v3_geo_ctx.py --geo "$GEO" --layer "$LAY" "$@" ;;
  index)   shift; $PY scripts/ner2_layer_geocode.py --forms "$GEO/forms.parquet" --doc-forms "$GEO/doc_forms.parquet" --out "$GEO/gc" --index-only "$@" ;;
  geocode) shift; $PY scripts/ner2_layer_geocode.py --forms "$GEO/forms.parquet" --doc-forms "$GEO/doc_forms.parquet" --out "$GEO/gc" --ctx-map "$GEO/ctx_map.parquet" --doc-link "$GEO/doc_link.parquet" "$@" ;;
  points)  shift; [ -f "$GEO/words_en.txt" ] || cp "$D/ner/layer_v1/geo/words_en.txt" "$GEO/words_en.txt"
           $PY scripts/ner2_layer_geo_points.py --layer "$LAY" --geo "$GEO" --words "$GEO/words_en.txt" "$@" ;;
  tables)  shift; $PY scripts/ner2_layer_v3_tables.py --v3 "$V3" --geo-name "${GEO_NAME:-geo}" --out "$V3/${SUMMARY:-summary}" "$@" ;;
  *) echo "unknown step $1" >&2; return 2 ;;
esac
}
if [ "$STEP" = all ]; then for s in merge prep ctx index geocode points; do run $s || exit $?; done; else run "$STEP" "$@"; fi
rc=$?; echo "[ner_v3 $STEP] done rc=$rc $(date +%T)"; exit $rc
