#!/bin/bash
# GeoNames name index for the geocoder (needs geonames/allCountries.zip already on Hypatia):
#   sbatch --output=$HOME/renascor-atlas-data/logs/%x-%j.out slurm/geocode_index.sbatch
#SBATCH --job-name=ratlas-gcindex
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=01:30:00
set -euo pipefail
D="$HOME/renascor-atlas-data"
mkdir -p "$D/geocode"
cd "$HOME/renascor-atlas"
nice -n 10 .venv/bin/python -I -m ratlas reconcile geocode index --local "$D/out/all/places_local.parquet" \
  --zip "$D/geonames/allCountries.zip" --out "$D/geocode/gn_index.parquet"
