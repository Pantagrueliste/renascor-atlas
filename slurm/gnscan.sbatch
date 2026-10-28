#!/bin/bash
# v2: names of the GeoNames features behind the ids + same-name neighbours (gates):
#   sbatch --output=$HOME/renascor-atlas-data/logs/%x-%j.out slurm/gnscan.sbatch
# needs ~/renascor-atlas-data/reconcile_v2/ids.tsv (ratlas reconcile ids) and out_v2/all/places_local.parquet
#SBATCH --job-name=ratlas-gnscan
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=01:30:00
set -euo pipefail
D="$HOME/renascor-atlas-data"
mkdir -p "$D/reconcile_v2"
cd "$HOME/renascor-atlas"
nice -n 10 .venv/bin/python -I -m ratlas reconcile gnscan --ids "$D/reconcile_v2/ids.tsv" --local "$D/out_v2/all/places_local.parquet" \
  --zip "$D/geonames/allCountries.zip" --dest "$D/reconcile_v2"
