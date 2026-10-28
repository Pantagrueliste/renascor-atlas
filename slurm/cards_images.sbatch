#!/bin/bash
# Place cards part 2 (network, ~2500 sequential downloads at <= 1/s): sbatch --dependency=afterok:<wd job> slurm/cards_images.sbatch [args]
#SBATCH --job-name=ratlas-cards-img
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
set -euo pipefail
D="$HOME/renascor-atlas-data/cards"
cd "$HOME/renascor-atlas"
PYTHONPATH=src nice -n 10 python3 -m ratlas.cards_images --wd "$D/out/wd_places.parquet" --work "$D/work" \
  --img "$D/img" --tsv "$D/out/wd_images.tsv" "$@"
