#!/bin/bash
# Cavriana transfer test: 8-gram / edition scan of every training source of S-full and HM (+ the v1 dev), on CPU (debug partition).
#   sbatch -p debug -c 12 --mem=48G -t 01:00:00 -J cav-filter scripts/ner3_cav_filter_job.sh [--write]
#SBATCH -o /home/cag30/renascor-atlas-data/ner/cav_transfer/logs/%x-%j.out
set -uo pipefail
D="$HOME/renascor-atlas-data/ner"; C="$D/cav_transfer"; SIL="$D/silver/final"; AGG="$D/external/aggregate"
source "$D/venv_train/bin/activate"; cd "$HOME/renascor-atlas"; export PYTHONPATH="$HOME/renascor-atlas/src" PYTHONUNBUFFERED=1
MODE="${1:-}"; OUTD="$C/filter_dry"; [ "$MODE" = --write ] && OUTD="$C/data"
run() { python scripts/ner3_cav_filter.py --cav "$C/bench/bench_cavriana.jsonl" --out-dir "$OUTD/$1" --file "$1=$2" $MODE > "$OUTD/$1.log" 2>&1; echo "[$1] exit $?"; }
mkdir -p "$OUTD"
run ours "$D/data_v1/train.jsonl" &
run dev "$D/data_v1/dev.jsonl" &
run hist_gold "$AGG/hist_gold/train.jsonl" &
run modern_silver "$AGG/modern_silver/train.jsonl" &
run silver_bal_es "$SIL/silver_balanced_es.jsonl" &
run silver_bal_it "$SIL/silver_balanced_it.jsonl" &
run silver_bal_la "$SIL/silver_balanced_la.jsonl" &
run silver_res_es "$SIL/silver_reserve_es.jsonl" &
run silver_res_it "$SIL/silver_reserve_it.jsonl" &
wait
grep -h "" "$OUTD"/*.log
