#!/bin/bash
# Cavriana transfer test: the runs of scripts/ner3_cav_runs.tsv in ONE GPU job, one task per run and per GPU.
#   sbatch -p gpu.L40S -N 1 --ntasks-per-node=2 --gres=gpu:2 -c 8 --mem=160G -t 04:00:00 -J cav-train scripts/ner3_cav_train_multi.sh scripts/ner3_cav_runs.tsv
# Resubmit the same command if the time limit stops a run: it resumes from its checkpoint.
#SBATCH -o /home/cag30/renascor-atlas-data/ner/cav_transfer/logs/%x-%j.out
set -uo pipefail
D="$HOME/renascor-atlas-data/ner"
export TV="$D/cav_transfer/train" RUNS="$(readlink -f "$1")"
export JD="$TV/jobs/$SLURM_JOB_ID"
mkdir -p "$JD" "$TV/runs" "$TV/models" "$TV/run_logs"
export JOB_END=$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e | head -1)" +%s)
cp "$RUNS" "$JD/runs.tsv"
echo "[cav] $(date +%T) job $SLURM_JOB_ID nodes ${SLURM_JOB_NODELIST:-?} end $(date -d @$JOB_END +%T)"
srun --kill-on-bad-exit=0 bash "$HOME/renascor-atlas/scripts/ner3_cav_train_task.sh"
echo "[cav] $(date +%T) all tasks finished"
grep -h "" "$JD"/task*.log
