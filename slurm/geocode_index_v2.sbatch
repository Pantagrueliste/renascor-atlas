#!/bin/bash
# v2: GeoNames name index with the new keys (small letters, skeleton folds, no common nouns):
#   sbatch --output=$HOME/renascor-atlas-data/logs/%x-%j.out slurm/geocode_index_v2.sbatch
#SBATCH --job-name=ratlas-gcindex2
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=01:30:00
set -euo pipefail
D="$HOME/renascor-atlas-data"
mkdir -p "$D/geocode_v2"
cd "$HOME/renascor-atlas"
nice -n 10 .venv/bin/python -I -m ratlas reconcile geocode index --local "$D/out_v2/all/places_local.parquet" \
  --zip "$D/geonames/allCountries.zip" --out "$D/geocode_v2/gn_index.parquet"
