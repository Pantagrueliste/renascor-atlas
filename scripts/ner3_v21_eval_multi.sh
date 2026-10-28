#!/bin/bash
# RenNER v2.1 evaluation in ONE Slurm job, one task per GPU: predictions (scripts/ner_audit_predict.py) of the given models on dev3,
# test_v2, the whole editor bench and the synthetic lowercase copy of bench-dev (train_v21/bench_dev/bench_dev_lc.jsonl).
# Outputs with a non-empty file are skipped (resumable). Gold v2 and gold v3 are never read.
#   sbatch -J r5-eval -p gpu.L40S --exclude=gpu03 -N 1 --ntasks-per-node=2 --gres=gpu:2 -c 8 --mem=120G -t 01:00:00 \
#     -o ~/renascor-atlas-data/ner/train_v21/eval/logs/%x-%j.out scripts/ner3_v21_eval_multi.sh MODEL_DIR [MODEL_DIR ...]
set -uo pipefail
D="$HOME/renascor-atlas-data/ner"
export EV="$D/train_v21/eval" JD="$D/train_v21/eval/jobs/$SLURM_JOB_ID" MODELS="$*"
mkdir -p "$EV"/{pred_dev3,pred_testv2,pred_bench,pred_bench_dev_lc,logs} "$JD/claims"
echo "[r5e] $(date +%T) job $SLURM_JOB_ID nodes $SLURM_JOB_NODELIST tasks $SLURM_NTASKS models $MODELS"
srun --kill-on-bad-exit=0 bash "$HOME/renascor-atlas/scripts/ner3_v21_eval_task.sh"
echo "[r5e] $(date +%T) all tasks finished"
cat "$JD"/task*.log
