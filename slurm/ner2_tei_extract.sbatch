#!/bin/bash
# NER layer, TEI editions, step 1: running text + editor tags (CPU, debug partition).
#SBATCH --job-name=layer_tei_extract
#SBATCH --partition=debug
#SBATCH --cpus-per-task=22
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/home/cag30/renascor-atlas-data/ner/layer_tei/logs/extract-%j.out
set -uo pipefail
T="$HOME/renascor-atlas-data/ner/layer_tei"
M="$HOME/renascor-atlas-data/ner/layer_v1"
cd "$HOME/renascor-atlas"
PY="$HOME/renascor-atlas/.venv/bin/python"
rm -rf "$T/work_run"
$PY -I scripts/ner2_tei_extract.py cands --docs-all "$M/extract/docs_all.parquet" --db "$M/meta_in/corpus_master_FROZEN_2026-06-26.sqlite" --out "$T/cands.parquet"
$PY -I scripts/ner2_tei_extract.py run --archive "$T/archive_in" --cands "$T/cands.parquet" --meta "$M/meta_in" --out "$T/work" --procs "${PROCS:-22}" --only ${ONLY:-}
$PY -I scripts/ner2_tei_extract.py merge --cands "$T/cands.parquet" --out "$T/work"
