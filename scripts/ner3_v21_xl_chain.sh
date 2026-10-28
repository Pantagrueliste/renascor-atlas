#!/bin/bash
# RenNER v2.1: one 4 h block of the XL run v21_xlmrxl_$VARIANT on one A100 (resumes from its checkpoint), then, in the same job, the
# predictions of the finished model (dev3, test_v2, editor bench, bench-dev lowercase copy). At the end of the block, if the run or its
# predictions are not finished, the block submits the next one with --dependency=afterany on itself (so only one of these jobs runs at
# a time), unless 2 other GPU jobs of ours are already running or queued (rule: at most 2 GPU jobs at once): then it stops and the
# status file says so. Status: ~/renascor-atlas-data/ner/train_v21/STATUS_xl.txt. Start or resume by hand:
#   cd ~/renascor-atlas && sbatch --export=ALL,VARIANT=lc25 -J r5-xl-chain -t 04:00:00 -p gpu.A100 -N 1 --ntasks-per-node=1 --gres=gpu:1 \
#     -c 8 --mem=120G -o ~/renascor-atlas-data/ner/train_v21/logs/%x-%j.out scripts/ner3_v21_xl_chain.sh
# Stop the chain: touch ~/renascor-atlas-data/ner/train_v21/STOP_xl (then scancel the pending block if any).
set -uo pipefail
V="${VARIANT:-lc25}"; D="$HOME/renascor-atlas-data/ner"; TV="$D/train_v21"; RUN="v21_xlmrxl_$V"; ST="$TV/STATUS_xl.txt"
cd "$HOME/renascor-atlas"
export TV_DIR="$TV"
say() { echo "$(date '+%F %T') job $SLURM_JOB_ID: $*" | tee -a "$ST"; }
say "block start on $(hostname -s), run $RUN"
if [ ! -f "$TV/runs/$RUN/done.json" ]; then
  bash scripts/ner3_train_multi.sh "scripts/ner3_train_v21_runs_xl_$V.tsv"
  last=$(grep -E '^[0-9:]+ step [0-9]+/' "$TV/run_logs/$RUN.log" | tail -1 | awk '{print $3}')
  say "training block over: step $last; done=$([ -f "$TV/runs/$RUN/done.json" ] && echo yes || echo no)"
fi
EV="$TV/eval"
need_eval() { for s in dev3 testv2 bench bench_dev_lc; do [ -s "$EV/pred_$s/$RUN.jsonl.gz" ] || return 0; done; return 1; }
if [ -f "$TV/runs/$RUN/done.json" ] && need_eval; then
  left=$(( $(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e | head -1)" +%s) - $(date +%s) ))
  if [ "$left" -gt 1800 ]; then
    export EV JD="$EV/jobs/$SLURM_JOB_ID" MODELS="$TV/models/$RUN"
    mkdir -p "$JD/claims" "$EV/logs"
    srun -n1 bash scripts/ner3_v21_eval_task.sh
    say "predictions: $(for s in dev3 testv2 bench bench_dev_lc; do [ -s "$EV/pred_$s/$RUN.jsonl.gz" ] && echo -n "$s ok; " || echo -n "$s MISSING; "; done)"
  else
    say "run done, $left s left: predictions left to the next block"
  fi
fi
if [ -f "$TV/runs/$RUN/done.json" ] && ! need_eval; then
  say "ALL DONE: model $TV/models/$RUN, predictions in $EV/pred_*/$RUN.jsonl.gz"; exit 0
fi
[ -f "$TV/STOP_xl" ] && { say "STOP_xl present: chain stopped"; exit 0; }
others=$(squeue -h -u "$USER" -t R,PD -o "%i %P" | grep -i gpu | grep -v "^$SLURM_JOB_ID " | wc -l)
if [ "$others" -ge 2 ]; then
  say "NOT resubmitted: $others other GPU jobs of ours queued or running. Resume by hand (command in the header of scripts/ner3_v21_xl_chain.sh)"
  exit 0
fi
nj=$(sbatch --parsable --dependency=afterany:"$SLURM_JOB_ID" --export=ALL,VARIANT="$V" -J r5-xl-chain -t 04:00:00 -p gpu.A100 -N 1 \
     --ntasks-per-node=1 --gres=gpu:1 -c 8 --mem=120G -o "$TV/logs/%x-%j.out" scripts/ner3_v21_xl_chain.sh)
say "next block submitted: $nj (starts after $SLURM_JOB_ID)"
