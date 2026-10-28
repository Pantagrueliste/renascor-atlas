#!/bin/bash
# One task of scripts/ner3_seeds_train_multi.sh. Same as scripts/ner3_train_task.sh (recipe unchanged), except: the seed comes from the last
# column of $RUNS, the dev3 / test_v2 / editor-bench predictions follow each finished run, and the CPU count follows the allocation.
set -uo pipefail
set -f
D="$HOME/renascor-atlas-data/ner"
s=$(printf "%02d" "$SLURM_PROCID")
LOG="$JD/task$s.log"
nvidia-smi --query-gpu=timestamp,pci.bus_id,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,noheader -l 60 > "$JD/gpu_task$s.csv" 2>&1 &
GPULOG=$!
trap 'kill $GPULOG 2>/dev/null' EXIT
export CUDA_VISIBLE_DEVICES=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d, -f$((SLURM_LOCALID + 1)))
case "$(hostname -s)" in gpu04*|gpu05*|gpu06*) BIG=1 ;; *) BIG=0 ;; esac
source "$D/venv_train/bin/activate"
cd "$HOME/renascor-atlas"
source scripts/ner_sets.sh
source scripts/ner_mixes.sh
[ -n "${R6_MIXES:-}" ] && source "$R6_MIXES"   # extra mixes (Latin v4: scripts/ner_mixes_lv4.sh)
DEV="${R6_DEV:-$OURS/dev.jsonl}"   # early-stopping dev set (variant without T3: a filtered copy)
NT=$(( ${SLURM_CPUS_PER_TASK:-8} > 4 ? 4 : ${SLURM_CPUS_PER_TASK:-8} ))
export PYTHONPATH="$HOME/renascor-atlas/src" HF_HUB_CACHE="$D/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true OMP_NUM_THREADS=$NT PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MARGIN=10
EVALS="$TRAIN_EVAL_ARGS --save-preds test"
EV="$TV/eval"
predict() {   # run name
  local M="$1" job IN OUT t0
  for job in dev3 testv2 bench; do
    case "$job" in
      dev3)   IN="$D/v3/dev3/passages.jsonl"; OUT="$EV/pred_dev3/$M.jsonl.gz" ;;
      testv2) IN="$D/v2/test2/test_v2_loc.jsonl"; OUT="$EV/pred_testv2/$M.jsonl.gz" ;;
      bench)  IN="$D/v2/editor_bench/bench.jsonl"; OUT="$EV/pred_bench/$M.jsonl.gz" ;;
    esac
    [ -s "$OUT" ] && continue
    t0=$(date +%s)
    TOKENIZERS_PARALLELISM=false python scripts/ner_audit_predict.py --model-dir "$TV/models/$M" --data "$IN" --out "$OUT.part" --words-from tokens \
        > "$EV/logs/$M.$job.log" 2>&1 && mv "$OUT.part" "$OUT"
    echo "[task $s] $(date "+%F %T") $M $job rc=$? wall=$(( $(date +%s) - t0 ))s" >> "$LOG"
  done
}
echo "[task $s] $(date "+%F %T") $(hostname) gpu $CUDA_VISIBLE_DEVICES big $BIG cpus ${SLURM_CPUS_PER_TASK:-?} $(nvidia-smi --query-gpu=name,pci.bus_id,memory.total --format=csv,noheader -i "$CUDA_VISIBLE_DEVICES" 2>/dev/null)" >> "$LOG"
[ "$BIG" = 0 ] && sleep 60
while IFS='|' read -r name mixv model lr minstart gpu extra extra_small seed; do
  [ -z "$name" ] && continue
  case "$name" in \#*) continue ;; esac
  [ -f "$TV/runs/$name/done.json" ] && continue
  [ "$gpu" = a100 ] && [ "$BIG" = 0 ] && continue
  [ "$BIG" = 0 ] && [ -n "$extra_small" ] && extra="$extra_small"
  mkdir "$JD/claims/$name" 2>/dev/null || continue
  left=$(( (JOB_END - $(date +%s)) / 60 - MARGIN ))
  if [ "$left" -lt "$minstart" ]; then
    echo "[task $s] $(date "+%F %T") $name: $left min left < $minstart, not started" >> "$LOG"
    rmdir "$JD/claims/$name"; continue
  fi
  echo "$(hostname) gpu $CUDA_VISIBLE_DEVICES task $s $(date "+%F %T")" > "$JD/claims/$name/who"
  echo "[task $s] $(date "+%F %T") start $name seed $seed ($left min)" >> "$LOG"
  echo "===== job $SLURM_JOB_ID task $s $(hostname) gpu $CUDA_VISIBLE_DEVICES $(date '+%F %T') limit $left min seed $seed args: $extra" >> "$TV/run_logs/$name.log"
  # the training limit leaves ~20 min for the predictions
  nice -n 5 python -m ratlas.ner.train --name "$name" --model "$model" --lr "$lr" $extra ${!mixv} --dev "$DEV" $EVALS \
      --out "$TV/runs" --models "$TV/models" --cache "$ND/cache" --max-len 256 --seed "$seed" --time-limit-min "$(( left - 20 ))" < /dev/null >> "$TV/run_logs/$name.log" 2>&1
  rc=$?
  echo "[task $s] $(date "+%F %T") $name exit $rc" >> "$LOG"
  echo "$rc" > "$JD/claims/$name/exit"
  [ -f "$TV/runs/$name/done.json" ] && predict "$name"
  [ "$rc" = 3 ] && break     # time limit: the job is ending
done < "$JD/runs.tsv"
echo "[task $s] $(date "+%F %T") no run left" >> "$LOG"
exit 0
