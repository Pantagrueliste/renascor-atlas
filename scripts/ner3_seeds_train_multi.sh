#!/bin/bash
# RenNER seeds 2 and 3 in ONE Slurm job (copy of scripts/ner3_train_multi.sh, which the XL v2.1 chain uses and which stays untouched):
# one run per task, one GPU per task; every task claims the next run of the list, trains it (seed from the last column), then predicts
# dev3, test_v2 and the whole editor bench with the finished model (scripts/ner_audit_predict.py). Gold v2 and gold v3 are never read.
#   sbatch -J r6-seeds -t 04:00:00 -o ~/renascor-atlas-data/ner/train_seeds/logs/%x-%j.out \
#     -p gpu.A100 -w gpu05 -N 1 --ntasks-per-node=2 --gres=gpu:2 -c 8 --mem=100G : \
#     -p gpu.L40S --exclude=gpu03 -N 1 --ntasks-per-node=4 --gres=gpu:4 -c 4 --mem=64G  scripts/ner3_seeds_train_multi.sh scripts/ner3_seeds_runs.tsv
set -uo pipefail
D="$HOME/renascor-atlas-data/ner"
export TV="${TV_DIR:-$D/train_seeds}" RUNS="$(readlink -f "$1")"
export JD="$TV/jobs/$SLURM_JOB_ID"
mkdir -p "$JD/claims" "$TV/runs" "$TV/models" "$TV/run_logs" "$TV/logs" "$TV/eval"/{pred_dev3,pred_testv2,pred_bench,logs}
export JOB_END=$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e | head -1)" +%s)
cp "$RUNS" "$JD/runs.tsv"
HET=""
if [ -n "${SLURM_HET_SIZE:-}" ]; then
  HET="--het-group=0-$((SLURM_HET_SIZE - 1))"
  for ((g = 0; g < SLURM_HET_SIZE; g++)); do
    v="SLURM_JOB_NODELIST_HET_GROUP_$g"; t="SLURM_NTASKS_HET_GROUP_$g"; p="SLURM_JOB_PARTITION_HET_GROUP_$g"
    echo "[r6] component $g: ${!p:-?} ${!v:-?} ${!t:-?} tasks"
  done
fi
echo "[r6] $(date +%T) job $SLURM_JOB_ID nodes ${SLURM_JOB_NODELIST:-?} end $(date -d @$JOB_END +%T) $HET"
srun $HET --kill-on-bad-exit=0 bash "$HOME/renascor-atlas/scripts/ner3_seeds_train_task.sh"
echo "[r6] $(date +%T) all tasks finished"
grep -h "" "$JD"/task*.log
