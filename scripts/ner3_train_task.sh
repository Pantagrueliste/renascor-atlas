#!/bin/bash
# One task of scripts/ner3_train_multi.sh: the GPU of index SLURM_LOCALID on its node; claims runs of $RUNS in order (mkdir is atomic on GPFS).
# A run marked "a100" is only claimed on an A100 node (gpu04-06, 80 GB); on a smaller GPU (L40S, 48 GB) the run takes its last column of
# arguments when it has one. Tasks on smaller GPUs wait 60 s before their first claim, so that the A100 tasks take the first runs of the list.
set -uo pipefail
set -f   # the arguments hold glob patterns for the Python side
D="$HOME/renascor-atlas-data/ner"
s=$(printf "%02d" "$SLURM_PROCID")
LOG="$JD/task$s.log"
# GPU logger, once a minute: every GPU of the job on this node (bus id tells them apart)
nvidia-smi --query-gpu=timestamp,pci.bus_id,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,noheader -l 60 > "$JD/gpu_task$s.csv" 2>&1 &
GPULOG=$!
trap 'kill $GPULOG 2>/dev/null' EXIT
export CUDA_VISIBLE_DEVICES=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d, -f$((SLURM_LOCALID + 1)))
case "$(hostname -s)" in gpu04*|gpu05*|gpu06*) BIG=1 ;; *) BIG=0 ;; esac
source "$D/venv_train/bin/activate"
cd "$HOME/renascor-atlas"
source scripts/ner_sets.sh
source scripts/ner_mixes.sh
export PYTHONPATH="$HOME/renascor-atlas/src" HF_HUB_CACHE="$D/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MARGIN=10
EVALS="$TRAIN_EVAL_ARGS --save-preds test"   # as v1; dev3 is scored afterwards with its own pipeline (alternative spans)
echo "[task $s] $(date "+%F %T") $(hostname) gpu $CUDA_VISIBLE_DEVICES big $BIG $(nvidia-smi --query-gpu=name,pci.bus_id,memory.total --format=csv,noheader -i "$CUDA_VISIBLE_DEVICES" 2>/dev/null)" >> "$LOG"
[ "$BIG" = 0 ] && sleep 60
while IFS='|' read -r name mixv model lr minstart gpu extra extra_small; do
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
  echo "[task $s] $(date "+%F %T") start $name ($left min)" >> "$LOG"
  echo "===== job $SLURM_JOB_ID task $s $(hostname) gpu $CUDA_VISIBLE_DEVICES $(date '+%F %T') limit $left min args: $extra" >> "$TV/run_logs/$name.log"
  nice -n 5 python -m ratlas.ner.train --name "$name" --model "$model" --lr "$lr" $extra ${!mixv} --dev "$OURS/dev.jsonl" $EVALS \
      --out "$TV/runs" --models "$TV/models" --cache "$ND/cache" --max-len 256 --seed 1 --time-limit-min "$left" < /dev/null >> "$TV/run_logs/$name.log" 2>&1
  rc=$?
  echo "[task $s] $(date "+%F %T") $name exit $rc" >> "$LOG"
  echo "$rc" > "$JD/claims/$name/exit"
  [ "$rc" = 3 ] && break     # time limit: the job is ending
done < "$JD/runs.tsv"
echo "[task $s] $(date "+%F %T") no run left" >> "$LOG"
exit 0
