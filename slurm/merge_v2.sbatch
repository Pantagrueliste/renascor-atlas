#!/bin/bash
# v2 merge on a compute node, after the extraction array:
#   sbatch --output=$HOME/renascor-atlas-data/logs/%x-%j.out [--dependency=afterok:<array job id>] slurm/merge_v2.sbatch
# Reads ~/renascor-atlas-data/out_v2/<slug>/, writes ~/renascor-atlas-data/out_v2/all/ (mentions, entities, collections.tsv, deviations*.tsv,
# counts_by_zone_class.tsv, *_local.parquet) and compares with the v1 collections table (v1_collections.tsv, synced from RQ/derived/atlas/v1).
#SBATCH --job-name=ratlas-merge-v2
#SBATCH --partition=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:30:00
set -euo pipefail
export RATLAS_DATA="$HOME/renascor-atlas-data"
export RATLAS_RAW="$RATLAS_DATA/raw"
export RATLAS_SCAN="$RATLAS_DATA/scan.jsonl.gz"
export RATLAS_SCAN_EXTRA="$RATLAS_DATA/scan_extra.jsonl.gz"
export RATLAS_OUT="$RATLAS_DATA/out_v2"
cd "$HOME/renascor-atlas"
nice -n 10 .venv/bin/python -I -m ratlas merge --out-root "$RATLAS_OUT" --dest "$RATLAS_OUT/all" --v1-collections "$RATLAS_DATA/v1_collections.tsv" "$@"
