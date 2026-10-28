#!/bin/bash
# NER layer, TEI editions, steps 2-3: duplicate control against the model layer and inside the set, seen-by-v1 share, final tables and chunks (CPU, debug partition).
#SBATCH --job-name=layer_tei_dedup
#SBATCH --partition=debug
#SBATCH --cpus-per-task=22
#SBATCH --mem=128G
#SBATCH --time=01:45:00
#SBATCH --output=/home/cag30/renascor-atlas-data/ner/layer_tei/logs/dedup-%j.out
set -uo pipefail
N="$HOME/renascor-atlas-data/ner"
T="$N/layer_tei"
cd "$HOME/renascor-atlas"
PY="$HOME/renascor-atlas/.venv/bin/python"
P="${PROCS:-22}"
$PY -I scripts/ner2_tei_extract.py merge --cands "$T/cands.parquet" --out "$T/work"
$PY -I scripts/ner2_tei_dedup.py index --work "$T/work" --procs $P
$PY -I scripts/ner2_tei_dedup.py scan --work "$T/work" --name layer --procs $P --glob "$N/layer_v1/extract/texts/[a-z][a-z]/*.jsonl.gz"
$PY -I scripts/ner2_tei_dedup.py scan --work "$T/work" --name train --procs $P --glob "$N/data_v1/train.jsonl" "$N/data_v1/dev.jsonl" "$N/data_v1/test.jsonl" "$N/gold_seed_v1.jsonl" "$N/external/aggregate/hist_gold/train.jsonl" "$N/external/aggregate/hist_gold/dev.jsonl" "$N/external/aggregate/hist_gold/test/*.jsonl"
$PY -I scripts/ner2_tei_dedup.py within --work "$T/work"
$PY -I scripts/ner2_tei_dedup.py decide --work "$T/work"
$PY -I scripts/ner2_tei_assemble.py --work "$T/work" --out "$T"
