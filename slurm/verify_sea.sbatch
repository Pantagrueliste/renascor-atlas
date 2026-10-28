#!/bin/bash
# Distance of every name-based point to the nearest populated place of GeoNames (allCountries.zip already on Hypatia):
#   sbatch --output=$HOME/renascor-atlas-data/logs/%x-%j.out slurm/verify_sea.sbatch
#SBATCH --job-name=ratlas-vsea
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:40:00
set -euo pipefail
D="$HOME/renascor-atlas-data"
cd "$HOME/renascor-atlas"
nice -n 10 .venv/bin/python -I -m ratlas reconcile verify sea --geo "$D/out/reconcile_v1.1/places_local_geo.parquet" \
  --zip "$D/geonames/allCountries.zip" --out "$D/geocode/sea_check.parquet"
