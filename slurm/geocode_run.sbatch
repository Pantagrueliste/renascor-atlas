#!/bin/bash
# Name-based geocoding + precision self-check (after geocode_index):
#   sbatch --output=$HOME/renascor-atlas-data/logs/%x-%j.out slurm/geocode_run.sbatch [extra args]
#SBATCH --job-name=ratlas-geocode
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
D="$HOME/renascor-atlas-data"
cd "$HOME/renascor-atlas"
nice -n 10 .venv/bin/python -I -m ratlas reconcile geocode run --geo "$D/out/reconcile_v1/places_local_geo.parquet" \
  --local "$D/out/all/places_local.parquet" --mentions "$D/out/all/mentions.parquet" --index "$D/geocode/gn_index.parquet" \
  --out "$D/geocode" "$@"
