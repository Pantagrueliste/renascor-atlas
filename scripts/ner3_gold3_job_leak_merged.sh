#!/bin/bash
#SBATCH --job-name=g3m-leak
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --output=%x-%j.out
# gold v3 merged with the "retrait du silver" option (2026-10-24; Hypatia debug, CPU only). The leak checks of ner3_gold3_job_leak.sh and
# ner3_gold3_job_docleak.sh, re-run on the merged set:
#   1. new passages (gold3/...) : ner2_leak_check.py against v1 train/dev/test, the mixes, test_v2, gold v2 and the seed (flag 0.02, as dev3);
#   2. gold v2 passages          : same, without gold v2 and the seed (they are those texts);
#   3. documents of the new passages, whole text: ner3_gold3_doc_leak.py against v1 and the mixes (25 8-grams or 1 %).
set -uo pipefail
D="$HOME/renascor-atlas-data/ner"; W="$D/v2/gold3/merged"; PY="$HOME/renascor-atlas/.venv/bin/python"; cd "$HOME/renascor-atlas"
A="$D/external/aggregate"
V1MIX=("$D/data_v1/train.jsonl" "$D/data_v1/dev.jsonl" "$D/data_v1/test.jsonl" "$A/hist_gold/train.jsonl" "$A/hist_gold/dev.jsonl" "$A/hist_gold/test/*.jsonl"
       "$A/all_private/train.jsonl" "$A/hist_open/train.jsonl" "$A/modern_silver/train.jsonl")
rc=0
nice -n 10 $PY scripts/ner2_leak_check.py --passages "$W/leak_in_new.jsonl" --out "$W/leak_remote_new.json" --flag 0.02 \
  --stream "${V1MIX[@]}" "$D/v2/test2/test_v2.jsonl" "$D/v2/gold2/passages.jsonl" "$D/gold_seed_v1.jsonl" || rc=1
nice -n 10 $PY scripts/ner2_leak_check.py --passages "$W/leak_in_g2.jsonl" --out "$W/leak_remote_g2.json" --flag 0.02 \
  --stream "${V1MIX[@]}" "$D/v2/test2/test_v2.jsonl" || rc=1
nice -n 10 $PY "$D/v2/gold3/ner3_gold3_doc_leak.py" --docs "$W/merged_docs.jsonl.gz" --out "$W/doc_leak_merged.json" --stream "${V1MIX[@]}" || rc=1
exit $rc
