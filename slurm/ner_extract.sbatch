#!/bin/bash
# NER data, step 1: the passages of ONE edition (array task = one slug). Do not sbatch this directly: use scripts/ner_submit.sh slug...
# Debug partition, 8 cpus per task, at most 3 tasks at once (24 cpus), 24G, 1 h 30. CPU only.
#SBATCH --job-name=ratlas-ner-extract
#SBATCH --partition=debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:30:00
set -euo pipefail
: "${SLUGS:?SLUGS (space-separated) must be exported, use scripts/ner_submit.sh}"
read -ra ARR <<< "$SLUGS"
SLUG="${ARR[$SLURM_ARRAY_TASK_ID]}"
D="$HOME/renascor-atlas-data"
export RATLAS_DATA="$D" RATLAS_RAW="$D/raw" RATLAS_SCAN="$D/scan.jsonl.gz" RATLAS_SCAN_EXTRA="$D/scan_extra.jsonl.gz"
OUT="$D/ner/${NER_VERSION:-data_v1}/raw"
mkdir -p "$OUT"
cd "$HOME/renascor-atlas"
echo "ner extract $SLUG on $(hostname) with ${SLURM_CPUS_PER_TASK:-1} cpus -> $OUT"
nice -n 10 .venv/bin/python -I -m ratlas.ner.extract "$SLUG" --out "$OUT" --meta "$D/ner/meta" --workers "${SLURM_CPUS_PER_TASK:-1}" ${LIMIT:+--limit "$LIMIT"}
