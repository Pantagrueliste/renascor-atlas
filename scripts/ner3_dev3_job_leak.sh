#!/bin/bash
# dev3 job (Hypatia debug, CPU): 8-gram leak check of the candidate passages against v1 train/dev/test, the external aggregates, test_v2, gold v2 and the seed.
set -uo pipefail
D="$HOME/renascor-atlas-data/ner"; W="$D/v2/dev3"; PY="$HOME/renascor-atlas/.venv/bin/python"; cd "$HOME/renascor-atlas"
A="$D/external/aggregate"
$PY scripts/ner2_leak_check.py --passages "$W/out/${CAND:-cand_dev3.jsonl}" --out "$W/out/${LEAK:-leak_cand_dev3.json}" --flag 0.02 --stream \
  "$D/data_v1/train.jsonl" "$D/data_v1/dev.jsonl" "$D/data_v1/test.jsonl" "$A/hist_gold/train.jsonl" "$A/hist_gold/dev.jsonl" "$A/hist_gold/test/*.jsonl" \
  "$A/all_private/train.jsonl" "$A/modern_silver/train.jsonl" "$D/v2/test2/test_v2.jsonl" "$D/v2/gold2/passages.jsonl" "$D/gold_seed_v1.jsonl"
