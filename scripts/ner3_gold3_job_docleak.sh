#!/bin/bash
#SBATCH --job-name=g3-docleak
#SBATCH --partition=debug
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --output=%x-%j.out
# gold v3 (Hypatia debug, CPU only): whole-document 8-gram check of the candidate documents against v1 and the training mixes.
set -uo pipefail
D="$HOME/renascor-atlas-data/ner"; W="$D/v2/gold3"; PY="$HOME/renascor-atlas/.venv/bin/python"; A="$D/external/aggregate"
nice -n 10 $PY "$W/ner3_gold3_doc_leak.py" --docs "$W/cand_docs.jsonl.gz" --out "$W/doc_leak.json" \
  --stream "$D/data_v1/train.jsonl" "$D/data_v1/dev.jsonl" "$D/data_v1/test.jsonl" "$A/hist_gold/train.jsonl" "$A/hist_gold/dev.jsonl" "$A/hist_gold/test/*.jsonl" \
  "$A/all_private/train.jsonl" "$A/hist_open/train.jsonl" "$A/modern_silver/train.jsonl"
