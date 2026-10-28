#!/bin/bash
# Full rebuild into a separate output root (verification run): extract all collections, merge, reconcile in two passes
# (pass 1 without namecheck, then `reconcile namecheck`, then pass 2 with the arbitration). Nothing under out/ is touched.
#   sbatch --output=$HOME/renascor-atlas-data/logs/%x-%j.out slurm/rebuild_all.sbatch [out-dir-name]
#SBATCH --job-name=ratlas-rebuild
#SBATCH --partition=debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
set -euo pipefail
NAME="${1:-out_verify}"
export RATLAS_DATA="$HOME/renascor-atlas-data"
export RATLAS_RAW="$RATLAS_DATA/raw"
export RATLAS_SCAN="$RATLAS_DATA/scan.jsonl.gz"
export RATLAS_OUT="$RATLAS_DATA/$NAME"
D="$RATLAS_DATA"
mkdir -p "$RATLAS_OUT"
cd "$HOME/renascor-atlas"
PY="nice -n 10 .venv/bin/python -I"
SLUGS=$(ls src/ratlas/collections/*.py | xargs -n1 basename | sed 's/\.py$//' | grep -v -e '^__init__$' -e '^generic_tei$')
for s in $SLUGS; do
  echo "== extract $s $(date +%T)"
  $PY -m ratlas extract "$s" --workers "${SLURM_CPUS_PER_TASK:-1}" 2>&1 | tail -3
done
echo "== merge $(date +%T)"
$PY -m ratlas merge --out-root "$RATLAS_OUT" --dest "$RATLAS_OUT/all"
# reconcile work files: a private copy of the small tables + overrides
RD="$D/reconcile_$NAME"
rm -rf "$RD"; mkdir -p "$RD"
for f in ids wd_map wd_places wd_persons geonames_lookup; do cp "$D/reconcile/$f.tsv" "$RD/"; done
[ -f "$D/reconcile/coord_overrides.tsv" ] && cp "$D/reconcile/coord_overrides.tsv" "$RD/"
echo "== reconcile pass 1 $(date +%T)"
$PY -m ratlas reconcile build --v1 "$RATLAS_OUT/all" --rdir "$RD" --geonames "$RD/geonames_lookup.tsv" --dest "$RATLAS_OUT/reconcile_p1" > /dev/null
echo "== namecheck $(date +%T)"
$PY -m ratlas reconcile namecheck --geo "$RATLAS_OUT/reconcile_p1/places_local_geo.parquet" \
  --local "$RATLAS_OUT/all/places_local.parquet" --geonames-zip "$D/geonames/allCountries.zip" --dest "$RD/namecheck_local.tsv"
echo "== reconcile pass 2 $(date +%T)"
$PY -m ratlas reconcile build --v1 "$RATLAS_OUT/all" --rdir "$RD" --geonames "$RD/geonames_lookup.tsv" --dest "$RATLAS_OUT/reconcile_v1" > /dev/null
echo "== done $(date +%T)"
