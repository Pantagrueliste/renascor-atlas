#!/bin/bash
# v2: distance of the register / authority coordinates to a GeoNames feature of the same name (arbitration), on the pass-1 build:
#   sbatch --output=$HOME/renascor-atlas-data/logs/%x-%j.out slurm/namecheck_v2.sbatch
#SBATCH --job-name=ratlas-namecheck2
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=01:30:00
set -euo pipefail
D="$HOME/renascor-atlas-data"
cd "$HOME/renascor-atlas"
nice -n 10 .venv/bin/python -I -m ratlas reconcile namecheck --geo "$D/out_v2/reconcile/pass1/places_local_geo.parquet" \
  --local "$D/out_v2/all/places_local.parquet" --geonames-zip "$D/geonames/allCountries.zip" --dest "$D/reconcile_v2/namecheck_local.tsv"
