#!/bin/bash
# v2: audit rules on the build that has the geocoder results but no verify file -> geocode_verify.tsv
#   sbatch --output=$HOME/renascor-atlas-data/logs/%x-%j.out slurm/verify_v2.sbatch
#SBATCH --job-name=ratlas-verify2
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
D="$HOME/renascor-atlas-data"
cd "$HOME/renascor-atlas"
nice -n 10 .venv/bin/python -I -m ratlas reconcile verify rules --geo "$D/out_v2/reconcile/pre_verify/places_local_geo.parquet" --out "$D/geocode_v2/geocode_verify.tsv"
