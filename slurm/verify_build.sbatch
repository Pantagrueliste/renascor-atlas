#!/bin/bash
# Audit rules -> geocode_verify.tsv, then the reconcile build with the audited name-based coordinates:
#   sbatch --output=$HOME/renascor-atlas-data/logs/%x-%j.out slurm/verify_build.sbatch
# The unaudited tables stay in out/reconcile_v1.1; the audited ones go to out/reconcile_v1.1_verified.
#SBATCH --job-name=ratlas-vbuild
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:00:00
set -euo pipefail
D="$HOME/renascor-atlas-data"
cd "$HOME/renascor-atlas"
nice -n 10 .venv/bin/python -I -m ratlas reconcile verify rules --geo "$D/out/reconcile_v1.1/places_local_geo.parquet" \
  --out "$D/geocode/geocode_verify.tsv"
nice -n 10 .venv/bin/python -I -m ratlas reconcile build --v1 "$D/out/all" --rdir "$D/reconcile" \
  --geonames "$D/reconcile/geonames_lookup.tsv" --geocode "$D/geocode/geocode_places.parquet" \
  --geocode-verify "$D/geocode/geocode_verify.tsv" --dest "$D/out/reconcile_v1.1_verified" "$@"
