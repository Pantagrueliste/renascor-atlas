#!/bin/bash
# NER data, steps 2 and 3, after slurm/ner_extract.sbatch has finished for all editions:
#   sbatch --export=ALL,STEP=complete slurm/ner_build.sbatch     completeness of every edition (parallel over editions)
#   sbatch --export=ALL,STEP=splits   slurm/ner_build.sbatch     train / dev / test, JSONL, tables, DATASHEET.md
#   sbatch --export=ALL,STEP=pytest   slurm/ner_build.sbatch     the NER tests on the cluster
# Debug partition, 8 cpus, 96G, 1 h 30. CPU only.
#SBATCH --job-name=ratlas-ner-build
#SBATCH --partition=debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
set -euo pipefail
STEP="${STEP:?STEP=complete|splits|pytest}"
D="$HOME/renascor-atlas-data"
V="$D/ner/${NER_VERSION:-data_v1}"
export RATLAS_DATA="$D" RATLAS_RAW="$D/raw" RATLAS_SCAN="$D/scan.jsonl.gz" RATLAS_SCAN_EXTRA="$D/scan_extra.jsonl.gz"
cd "$HOME/renascor-atlas"
echo "[ner $STEP] $(date +%T) on $(hostname) with ${SLURM_CPUS_PER_TASK:-1} cpus"
case "$STEP" in
  complete) nice -n 10 .venv/bin/python -I -m ratlas.ner.complete --data "$V" --workers "${SLURM_CPUS_PER_TASK:-1}" ;;
  splits)   nice -n 10 .venv/bin/python -I -m ratlas.ner.splits --data "$V" --meta "$D/ner/meta" ;;
  pytest)   nice -n 10 .venv/bin/python -I -m pytest tests/test_ner_extract.py tests/test_ner_complete.py -q -p no:cacheprovider ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
echo "[ner $STEP] done $(date +%T)"
