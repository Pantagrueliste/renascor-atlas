#!/bin/bash
# v2.1 final build: one step of the chain (see slurm/v21_chain.sh). Never run this on the login node; every step goes through Slurm (debug partition).
#   sbatch --output=$HOME/renascor-atlas-data/logs/%x-%j.out [--cpus-per-task N --mem M --time T] slurm/v21.sbatch STEP [extra args]
# STEP: merge | gnscan | build2 NAME [args] | namecheck | gcindex | gcrun [args] | verify | audit | audit_editions | pytest
# Paths: inputs  ~/renascor-atlas-data/out_v2_1/<slug>/ (extraction), reconcile_v2_1/ (authority tables + overrides + casebooks map), tgn/, geonames/
#        outputs ~/renascor-atlas-data/out_v2_1/{all,reconcile/<name>,audit_v2}, geocode_v2_1/
#SBATCH --job-name=ratlas-v21
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=01:30:00
set -uo pipefail
STEP="${1:?step}"; shift || true
D="$HOME/renascor-atlas-data"
OUT="$D/out_v2_1"; R="$D/reconcile_v2_1"; G="$D/geocode_v2_1"
export RATLAS_DATA="$D" RATLAS_RAW="$D/raw" RATLAS_SCAN="$D/scan.jsonl.gz" RATLAS_SCAN_EXTRA="$D/scan_extra.jsonl.gz" RATLAS_OUT="$OUT"
mkdir -p "$G" "$OUT/reconcile"
cd "$HOME/renascor-atlas"
PY="nice -n 10 .venv/bin/python -I"
echo "[v21 $STEP] $(date +%T) on $(hostname) with ${SLURM_CPUS_PER_TASK:-1} cpus"
case "$STEP" in
  merge)
    $PY -m ratlas merge --out-root "$OUT" --dest "$OUT/all" --v1-collections "$D/v1_collections.tsv" "$@" ;;
  gnscan)
    $PY -m ratlas reconcile gnscan --ids "$R/ids.tsv" --local "$OUT/all/places_local.parquet" --zip "$D/geonames/allCountries.zip" --dest "$R" ;;
  build2)
    NAME="${1:?name}"; shift
    $PY -m ratlas reconcile build2 --v2 "$OUT/all" --rdir "$R" --dest "$OUT/reconcile/$NAME" --tgn-dir "$D/tgn" --casebooks-map "$R/casebooks_place_map.tsv" "$@" ;;
  namecheck)
    $PY -m ratlas reconcile namecheck --geo "$OUT/reconcile/pass1/places_local_geo.parquet" --local "$OUT/all/places_local.parquet" \
      --geonames-zip "$D/geonames/allCountries.zip" --dest "$R/namecheck_local.tsv" ;;
  gcindex)
    $PY -m ratlas reconcile geocode index --local "$OUT/all/places_local.parquet" --zip "$D/geonames/allCountries.zip" --out "$G/gn_index.parquet" --tgn-dir "$D/tgn" ;;
  gcrun)
    $PY -m ratlas reconcile geocode run --geo "$OUT/reconcile/pass2/places_local_geo.parquet" --local "$OUT/all/places_local.parquet" \
      --mentions "$OUT/all/mentions.parquet" --index "$G/gn_index.parquet" --out "$G" "$@" ;;
  verify)
    $PY -m ratlas reconcile verify rules --geo "$OUT/reconcile/pre_verify/places_local_geo.parquet" --out "$G/geocode_verify.tsv" ;;
  audit)
    $PY -m ratlas audit v2 --all "$OUT/all" --final "$OUT/reconcile/final" --rdir "$R" --raw "$D/raw" --out "$OUT/audit_v2" \
      --geonames-zip "$D/geonames/allCountries.zip" --tgn-dir "$D/tgn" "$@"
    rc=$?; echo "audit exit code $rc"; exit $rc ;;
  audit_editions)
    $PY -m ratlas audit editions --all "$OUT/all" --final "$OUT/reconcile/final" --raw "$D/raw" --out "$OUT/audit_v2/editions" --rdir "$R" \
      --dist "$OUT/audit_v2/gn_final_dist.tsv" "$@" ;;
  pytest)
    # the whole suite, with the real-data tests (raw harvest, scan) and the nine checks on the v2.1 tables
    RATLAS_AUDIT_V2=1 RATLAS_AUDIT_ALL="$OUT/all" RATLAS_AUDIT_FINAL="$OUT/reconcile/final" RATLAS_AUDIT_RDIR="$R" RATLAS_AUDIT_RAW="$D/raw" \
      RATLAS_AUDIT_OUT="$OUT/audit_v2" RATLAS_AUDIT_TGN="$D/tgn" RATLAS_AUDIT_GEONAMES="$D/geonames/allCountries.zip" \
      nice -n 10 .venv/bin/python -I -m pytest tests -q -p no:cacheprovider "$@"
    rc=$?; echo "pytest exit code $rc"; exit $rc ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
rc=$?
echo "[v21 $STEP] done rc=$rc $(date +%T)"
exit $rc
