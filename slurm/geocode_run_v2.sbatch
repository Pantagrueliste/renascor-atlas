#!/bin/bash
# v2: name-based geocoding with the barriers, on the pass-2 build (arbitrated, gated):
#   sbatch --output=$HOME/renascor-atlas-data/logs/%x-%j.out slurm/geocode_run_v2.sbatch [--eval-n 1000 --eval-seeds 1,2]
#SBATCH --job-name=ratlas-geocode2
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
D="$HOME/renascor-atlas-data"
cd "$HOME/renascor-atlas"
nice -n 10 .venv/bin/python -I -m ratlas reconcile geocode run --geo "$D/out_v2/reconcile/pass2/places_local_geo.parquet" \
  --local "$D/out_v2/all/places_local.parquet" --mentions "$D/out_v2/all/mentions.parquet" --index "$D/geocode_v2/gn_index.parquet" \
  --out "$D/geocode_v2" "$@"
