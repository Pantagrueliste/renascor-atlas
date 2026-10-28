#!/bin/bash
# Document table, stage 'raw': one array task per slug (SLUGS exported by slurm/docs_raw.sh).
#SBATCH --job-name=ratlas-docraw
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
: "${SLUGS:?}"
read -ra ARR <<< "$SLUGS"
SLUG="${ARR[$SLURM_ARRAY_TASK_ID]}"
export RATLAS_DATA="$HOME/renascor-atlas-data"
export RATLAS_RAW="$RATLAS_DATA/raw"
export RATLAS_SCAN="$RATLAS_DATA/scan.jsonl.gz"
cd "$HOME/renascor-atlas"
D="${DOCS_DIR:-$RATLAS_DATA/docs_cards}"
echo "docs raw $SLUG on $(hostname) sample=${SAMPLE:-0}"
nice -n 10 .venv/bin/python -I -m ratlas.docs_table raw "$SLUG" --keys "$D/keys.parquet" --out "$D/raw${RAWSUFFIX:-}" --sample "${SAMPLE:-0}"
