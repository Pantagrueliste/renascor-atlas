#!/bin/bash
# Cavriana transfer test: one task = one run of $RUNS (name|mix variable|model|lr), on the GPU of index SLURM_LOCALID. Recipe of
# scripts/ner3_train_task.sh (v2_mdeberta_Sfull / r3_mdeberta_HM), with the mixes and the v1 dev without Cavriana (ner3_cav_mixes.sh).
set -uo pipefail
set -f
D="$HOME/renascor-atlas-data/ner"
s=$(printf "%02d" "$SLURM_PROCID")
LOG="$JD/task$s.log"
nvidia-smi --query-gpu=timestamp,pci.bus_id,utilization.gpu,memory.used,power.draw --format=csv,noheader -l 60 > "$JD/gpu_task$s.csv" 2>&1 &
GPULOG=$!; trap 'kill $GPULOG 2>/dev/null' EXIT
export CUDA_VISIBLE_DEVICES=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d, -f$((SLURM_LOCALID + 1)))
source "$D/venv_train/bin/activate"
cd "$HOME/renascor-atlas"
source scripts/ner_sets.sh; source scripts/ner_mixes.sh; source scripts/ner3_cav_mixes.sh
export PYTHONPATH="$HOME/renascor-atlas/src" HF_HUB_CACHE="$D/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
line=$(grep -v '^#' "$RUNS" | sed -n "$((SLURM_PROCID + 1))p")
[ -z "$line" ] && { echo "[task $s] no run" >> "$LOG"; exit 0; }
IFS='|' read -r name mixv model lr <<< "$line"
[ -f "$TV/runs/$name/done.json" ] && { echo "[task $s] $name done" >> "$LOG"; exit 0; }
left=$(( (JOB_END - $(date +%s)) / 60 - 10 ))
echo "[task $s] $(date "+%F %T") $(hostname) gpu $CUDA_VISIBLE_DEVICES start $name ($left min)" >> "$LOG"
nice -n 5 python -m ratlas.ner.train --name "$name" --model "$model" --lr "$lr" ${!mixv} --dev "$CAV_DEV" $TRAIN_EVAL_ARGS --save-preds test \
    --out "$TV/runs" --models "$TV/models" --cache "$ND/cache" --max-len 256 --seed 1 --time-limit-min "$left" < /dev/null >> "$TV/run_logs/$name.log" 2>&1
rc=$?
echo "[task $s] $(date "+%F %T") $(hostname) $name exit $rc" >> "$LOG"
