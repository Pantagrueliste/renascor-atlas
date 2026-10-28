#!/bin/bash
# Place cards part 1 (network, ~110 API calls at 1/s): sbatch --output=$HOME/renascor-atlas-data/logs/%x-%j.out slurm/cards_wikidata.sbatch
#SBATCH --job-name=ratlas-cards-wd
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:40:00
set -euo pipefail
D="$HOME/renascor-atlas-data/cards"
cd "$HOME/renascor-atlas"
PYTHONPATH=src nice -n 10 python3 -m ratlas.cards_wikidata --v1 "$D/in" --dest "$D/out" --cache "$D/cache/wikidata"
