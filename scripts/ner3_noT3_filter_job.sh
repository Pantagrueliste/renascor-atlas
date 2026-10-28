#!/bin/bash
# RenNER variant without CELT and CIDTC (T3): filtered copies of every training and early-stopping file of the lc25-lv4 recipe, on CPU.
# By edition, plus an 8-gram check against every CELT / CIDTC passage of the v1 extraction (scripts/ner3_noT3_filter.py).
# Inputs are only read; outputs go to ~/renascor-atlas-data/ner/train_noT3/data. Gold v2, gold v3, dev3, test_v2 and the bench are never read.
#   sbatch -p debug -c 16 --mem=120G -t 01:30:00 -J r8-noT3-filter -o ~/renascor-atlas-data/ner/train_noT3/logs/%x-%j.out scripts/ner3_noT3_filter_job.sh
set -euo pipefail
D="$HOME/renascor-atlas-data/ner"
source "$D/venv_train/bin/activate"
cd "$HOME/renascor-atlas"
export PYTHONPATH="$HOME/renascor-atlas/src" PYTHONUNBUFFERED=1
O="$D/train_noT3/data"; mkdir -p "$O"
S="$D/silver/final"; S4="$D/silver/final_v4"; AGG="$D/external/aggregate"; RAW="$D/data_v1/raw/passages"
python scripts/ner3_noT3_filter.py --procs 10 \
  --ref celt=$RAW/celt.jsonl.gz --ref cidtc=$RAW/cidtc.jsonl.gz \
  --ref-from $D/data_v1/train.jsonl --ref-from $D/data_v1/dev.jsonl --ref-from $D/data_v1/test.jsonl \
  --file ours=$D/train_v21/data/train_nomap.jsonl:$O/train_nomap_noT3.jsonl \
  --file dev=$D/data_v1/dev.jsonl:$O/dev_noT3.jsonl \
  --file hist_gold=$AGG/hist_gold/train.jsonl:$O/hist_gold_train_noT3.jsonl \
  --file modern_silver=$AGG/modern_silver/train.jsonl:$O/modern_silver_train_noT3.jsonl \
  --file silver_bal_es=$S/silver_balanced_es.jsonl:$O/silver_balanced_es_noT3.jsonl \
  --file silver_bal_it=$S/silver_balanced_it.jsonl:$O/silver_balanced_it_noT3.jsonl \
  --file silver_bal_la=$S4/silver_balanced_la.jsonl:$O/silver_balanced_la_v4_noT3.jsonl \
  --file silver_res_es=$S/silver_reserve_es.jsonl:$O/silver_reserve_es_noT3.jsonl \
  --file silver_res_it=$S/silver_reserve_it.jsonl:$O/silver_reserve_it_noT3.jsonl \
  --aug aug_lc=$D/train_v21/aug/aug_lc25_lv4.jsonl:$O/aug_lc25_lv4_noT3.jsonl \
  --details $O/dropped.jsonl --out $O/filter_report.json
ls -la "$O"; ( cd "$O" && md5sum *.jsonl )
echo "done $(date +%T)"
