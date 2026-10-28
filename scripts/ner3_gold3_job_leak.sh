#!/bin/bash
#SBATCH --job-name=g3-leak
#SBATCH --partition=debug
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
# gold v3 (Hypatia debug, CPU only): 8-gram leak check of the new candidate passages (selection + reserve) against the v1 data (train/dev/test),
# the training mixes (hist_gold, modern_silver, all_private, hist_open), test_v2, gold v2 and the seed. Same tool and threshold as dev3 (flag 0.02).
set -uo pipefail
D="$HOME/renascor-atlas-data/ner"; W="$D/v2/gold3"; PY="$HOME/renascor-atlas/.venv/bin/python"; cd "$HOME/renascor-atlas"
A="$D/external/aggregate"
nice -n 10 $PY scripts/ner2_leak_check.py --passages "$W/select_new_plus_reserve.jsonl" --out "$W/leak_remote.json" --flag 0.02 \
  --stream "$D/data_v1/train.jsonl" "$D/data_v1/dev.jsonl" "$D/data_v1/test.jsonl" \
  "$A/hist_gold/train.jsonl" "$A/hist_gold/dev.jsonl" "$A/hist_gold/test/*.jsonl" \
  "$A/all_private/train.jsonl" "$A/hist_open/train.jsonl" "$A/modern_silver/train.jsonl" \
  "$D/v2/test2/test_v2.jsonl" "$D/v2/gold2/passages.jsonl" "$D/gold_seed_v1.jsonl"
