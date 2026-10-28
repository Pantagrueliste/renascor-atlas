#!/bin/bash
# v2.2 trial of the geocoder: gcrun with the v2.2 code on the v2.1 pass-2 tables, written to scratch_v22/geocode (nothing of v2.1 is touched).
#SBATCH --job-name=ratlas-v22-trial-gc
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
D="$HOME/renascor-atlas-data"; S="$D/scratch_v22"
mkdir -p "$S/geocode"
cd "$HOME/renascor-atlas"
nice -n 10 .venv/bin/python -I -m ratlas reconcile geocode run --geo "$D/out_v2_1/reconcile/pass2/places_local_geo.parquet" --local "$D/out_v2_1/all/places_local.parquet" \
  --mentions "$D/out_v2_1/all/mentions.parquet" --index "$D/geocode_v2_1/gn_index.parquet" --out "$S/geocode"
