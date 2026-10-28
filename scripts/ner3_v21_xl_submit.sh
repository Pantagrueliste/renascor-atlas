#!/bin/bash
# RenNER v2.1: (re)submit the XL run v21_xlmrxl_<VARIANT> (default lc12) for one more 4 h block on one A100; it resumes from its own
# checkpoint (ratlas.ner.train). Run on the Hypatia login node (sbatch only). Refuses when the run is done, when one of our jobs already
# holds it, or when two of our GPU jobs are already running or pending (rule: at most 2 GPU jobs at once, all workflows together).
#   bash scripts/ner3_v21_xl_submit.sh [lc12|lc25|lc12n]
set -euo pipefail
V="${1:-lc12}"; D="$HOME/renascor-atlas-data/ner"; TV="$D/train_v21"; RUN="v21_xlmrxl_$V"
[ -f "$TV/runs/$RUN/done.json" ] && { echo "$RUN is done"; exit 0; }
squeue -h -u "$USER" -o "%j" | grep -q -E '^r5-train' && { echo "a r5-train job is already queued or running"; squeue -u "$USER"; exit 1; }
n=$(squeue -h -u "$USER" -o "%P" | grep -c -i gpu || true)
[ "$n" -ge 2 ] && { echo "$n of our GPU jobs already queued or running: wait"; squeue -u "$USER"; exit 1; }
cd "$HOME/renascor-atlas"
sbatch --export=ALL,TV_DIR="$TV" -J r5-train-xl -t 04:00:00 -o "$TV/logs/%x-%j.out" -p gpu.A100 -N 1 --ntasks-per-node=1 --gres=gpu:1 \
  -c 8 --mem=120G scripts/ner3_train_multi.sh "scripts/ner3_train_v21_runs_xl_$V.tsv"
