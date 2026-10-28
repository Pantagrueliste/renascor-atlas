#!/bin/bash
# Reconcile build on a compute node: sbatch --output=$HOME/renascor-atlas-data/logs/%x-%j.out slurm/reconcile_build.sbatch
#SBATCH --job-name=ratlas-reconcile
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:00:00
set -euo pipefail
D="$HOME/renascor-atlas-data"
cd "$HOME/renascor-atlas"
nice -n 10 .venv/bin/python -I -m ratlas reconcile build --v1 "$D/out/all" --rdir "$D/reconcile" \
  --geonames "$D/reconcile/geonames_lookup.tsv" --dest "$D/out/reconcile_v1" "$@"
