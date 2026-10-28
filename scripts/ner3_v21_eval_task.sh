#!/bin/bash
# One task of scripts/ner3_v21_eval_multi.sh: GPU of index SLURM_LOCALID; claims (model, set) units in order (mkdir is atomic).
set -uo pipefail
D="$HOME/renascor-atlas-data/ner"
s=$(printf "%02d" "$SLURM_PROCID")
LOG="$JD/task$s.log"
# EVAL_GPUS: explicit list of GPU indices for a step run with --overlap inside the training job (idle GPUs only)
if [ -n "${EVAL_GPUS:-}" ]; then
  export CUDA_VISIBLE_DEVICES=$(echo "$EVAL_GPUS" | cut -d, -f$((SLURM_LOCALID + 1)))
else
  export CUDA_VISIBLE_DEVICES=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d, -f$((SLURM_LOCALID + 1)))
fi
source "$D/venv_train/bin/activate"
cd "$HOME/renascor-atlas"
export PYTHONPATH="$HOME/renascor-atlas/src" HF_HUB_CACHE="$D/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
echo "[task $s] $(date "+%F %T") $(hostname) gpu $CUDA_VISIBLE_DEVICES $(nvidia-smi --query-gpu=name,pci.bus_id --format=csv,noheader -i "$CUDA_VISIBLE_DEVICES")" >> "$LOG"
for MD in $MODELS; do
  M=$(basename "$MD")
  for job in dev3 testv2 bench_dev_lc bench; do
    case "$job" in
      dev3)   IN="$D/v3/dev3/passages.jsonl"; OUT="$EV/pred_dev3/$M.jsonl.gz" ;;
      testv2) IN="$D/v2/test2/test_v2_loc.jsonl"; OUT="$EV/pred_testv2/$M.jsonl.gz" ;;
      bench_dev_lc) IN="$D/train_v21/bench_dev/bench_dev_lc.jsonl"; OUT="$EV/pred_bench_dev_lc/$M.jsonl.gz" ;;
      bench)  IN="$D/v2/editor_bench/bench.jsonl"; OUT="$EV/pred_bench/$M.jsonl.gz" ;;
    esac
    [ -s "$OUT" ] && continue
    mkdir "$JD/claims/$M.$job" 2>/dev/null || continue
    t0=$(date +%s)
    python scripts/ner_audit_predict.py --model-dir "$MD" --data "$IN" --out "$OUT.part" --words-from tokens > "$EV/logs/$M.$job.log" 2>&1 \
      && mv "$OUT.part" "$OUT"
    echo "[task $s] $(date "+%F %T") $M $job rc=$? wall=$(( $(date +%s) - t0 ))s" >> "$LOG"
  done
done
echo "[task $s] $(date "+%F %T") done" >> "$LOG"
