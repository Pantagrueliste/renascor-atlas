#!/bin/bash
# dev3 job (Hypatia debug partition, CPU): extract, list excluded documents, build candidates, leak check.
set -uo pipefail
D="$HOME/renascor-atlas-data/ner"; W="$D/v2/dev3"; PY="$HOME/renascor-atlas/.venv/bin/python"; cd "$HOME/renascor-atlas"
mkdir -p "$W/out"
if [ ! -d "$W/ta/salamanca" ]; then TARS="$W/tars" ROOT="$W/ta" bash scripts/ner3_dev3_extract.sh; fi
$PY scripts/ner3_dev3_docs.py "$W/out/excl_docs.txt" "$D/data_v1/train.jsonl" "$D/data_v1/dev.jsonl" "$D/data_v1/test.jsonl" "$D/v2/test2/test_v2.jsonl" "$D/v2/gold2/passages.jsonl" "$D/gold_seed_v1.jsonl"
cat "$W/gold2_final_docs.txt" >> "$W/out/excl_docs.txt"
$PY scripts/ner3_dev3_build.py --root "$W/ta" --excl-docs "$W/out/excl_docs.txt" --out "$W/out/cand_dev3.jsonl" --report "$W/out/cand_dev3_report.json" --per-lang "${PER_LANG:-156}" 2> "$W/out/build.err"
tail -25 "$W/out/build.err"
