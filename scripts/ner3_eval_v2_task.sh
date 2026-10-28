#!/bin/bash
# One task of scripts/ner3_eval_v2_multi.sh: the GPU of index SLURM_LOCALID; claims models in order (mkdir is atomic) and, per model,
# predicts dev3 and test_v2, then times dev3 x50. Outputs with a non-empty file are skipped (resumable).
set -uo pipefail
D="$HOME/renascor-atlas-data/ner"
s=$(printf "%02d" "$SLURM_PROCID")
LOG="$JD/task$s.log"
export CUDA_VISIBLE_DEVICES=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d, -f$((SLURM_LOCALID + 1)))
source "$D/venv_train/bin/activate"
cd "$HOME/renascor-atlas"
export PYTHONPATH="$HOME/renascor-atlas/src" HF_HUB_CACHE="$D/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
echo "[task $s] $(date "+%F %T") $(hostname) gpu $CUDA_VISIBLE_DEVICES $(nvidia-smi --query-gpu=name --format=csv,noheader -i "$CUDA_VISIBLE_DEVICES")" >> "$LOG"
# biggest first, so that the long ones start at once
for M in v2_xlmrxl_Sbal v2_xlmrxl_Sfull v2_eb2b_Sbal v2_eb2b_Sfull v2_eb610_Sbal v2_eb610_Sfull v2_xlmrl_Sbal v2_xlmrl_Sfull \
         v2_mdeberta_Sbal v2_mdeberta_Sfull r3_mdeberta_HM; do
  mkdir "$JD/claims/$M" 2>/dev/null || continue
  case "$M" in r3_*) MD="$D/models/$M" ;; *) MD="$D/train_v2/models/$M" ;; esac
  echo "[task $s] $(date "+%F %T") start $M" >> "$LOG"
  for job in dev3 testv2 speed; do
    case "$job" in
      dev3)   IN="$D/v3/dev3/passages.jsonl"; OUT="$EV/pred_dev3/$M.jsonl.gz" ;;
      testv2) IN="$D/v2/test2/test_v2_loc.jsonl"; OUT="$EV/pred_testv2/$M.jsonl.gz" ;;
      speed)  IN="$EV/speed/dev3_x50.jsonl"; OUT="$EV/speed/$M.a100.jsonl.gz" ;;
    esac
    [ "$M" = r3_mdeberta_HM ] && [ "$job" != speed ] && continue   # its dev3 / test_v2 predictions exist (same predictor)
    [ -s "$OUT" ] && continue
    t0=$(date +%s.%N)
    python scripts/ner_audit_predict.py --model-dir "$MD" --data "$IN" --out "$OUT" --words-from tokens > "$EV/logs/$M.$job.log" 2>&1
    rc=$?
    echo "[task $s] $(date "+%F %T") $M $job rc=$rc wall=$(echo "$(date +%s.%N) - $t0" | bc)" >> "$LOG"
  done
done
echo "[task $s] $(date "+%F %T") done" >> "$LOG"
