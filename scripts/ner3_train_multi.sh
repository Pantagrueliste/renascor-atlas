#!/bin/bash
# NER v2 (RenNER) training grid in ONE Slurm job: one run per task, one GPU per task, over several nodes. The job may be heterogeneous
# (one component per node type, e.g. A100 and L40S), so that every free GPU is used while it stays one job:
#   sbatch -J r4-train -t 04:00:00 -o ~/renascor-atlas-data/ner/train_v2/logs/%x-%j.out \
#     -p gpu.A100 -N 1 --ntasks-per-node=2 --gres=gpu:2 -c 8 --mem=200G : -p gpu.A100 -N 1 --ntasks-per-node=1 --gres=gpu:1 -c 8 --mem=100G : \
#     -p gpu.L40S -N 1 --ntasks-per-node=5 --gres=gpu:5 -c 8 --mem=200G  scripts/ner3_train_multi.sh scripts/ner3_train_v2_runs.tsv
# (a plain job works too: sbatch -p gpu.A100 -N 3 --ntasks-per-node=2 --gres=gpu:2 ...). Every task claims the next run of the list that it can
# run on its GPU and that is neither done nor taken by another task (scripts/ner3_train_task.sh), trains it with a time limit set from the
# end of the job, and claims the next one when its run is done. Resubmit (with as many tasks as runs left) until every run is done: each run
# resumes from its own checkpoint. Gold v2 and dev3 are never read (dev3 is scored afterwards, with its own pipeline).
#SBATCH -t 04:00:00
#SBATCH -o /home/cag30/renascor-atlas-data/ner/train_v2/logs/%x-%j.out
set -uo pipefail
D="$HOME/renascor-atlas-data/ner"
export TV="${TV_DIR:-$D/train_v2}" RUNS="$(readlink -f "$1")"
export JD="$TV/jobs/$SLURM_JOB_ID"
mkdir -p "$JD/claims" "$TV/runs" "$TV/models" "$TV/run_logs" "$TV/logs"
export JOB_END=$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e | head -1)" +%s)
cp "$RUNS" "$JD/runs.tsv"
HET=""
if [ -n "${SLURM_HET_SIZE:-}" ]; then
  HET="--het-group=0-$((SLURM_HET_SIZE - 1))"
  for ((g = 0; g < SLURM_HET_SIZE; g++)); do
    v="SLURM_JOB_NODELIST_HET_GROUP_$g"; t="SLURM_NTASKS_HET_GROUP_$g"; p="SLURM_JOB_PARTITION_HET_GROUP_$g"
    echo "[r4] component $g: ${!p:-?} ${!v:-?} ${!t:-?} tasks"
  done
fi
echo "[r4] $(date +%T) job $SLURM_JOB_ID nodes ${SLURM_JOB_NODELIST:-?} end $(date -d @$JOB_END +%T) $HET"
srun $HET --kill-on-bad-exit=0 bash "$HOME/renascor-atlas/scripts/ner3_train_task.sh"
echo "[r4] $(date +%T) all tasks finished"
grep -h "" "$JD"/task*.log
