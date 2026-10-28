#!/bin/bash
# Editor-tag bench on GPU (one Slurm job, one task per GPU, any number of nodes): predictions of the v2 students with the audit predictor
# (scripts/ner_audit_predict.py, bf16 autocast on GPU), whole bench in one pass per model. Models are claimed in order (mkdir is atomic);
# an output that exists is skipped (resumable). r3_mdeberta_HM is also run once on GPU under the name r3_mdeberta_HM_gpu, to measure the
# effect of bf16-GPU vs fp32-CPU (its CPU predictions stay the reference). Gold v2 is never read here.
#   sbatch -J eb-gpu -p gpu.L40S -N 1 --ntasks-per-node=8 --gres=gpu:8 -c 8 --mem=320G -t 01:00:00 scripts/ner_editor_bench_job_gpu.sh
#SBATCH -o /home/cag30/renascor-atlas-data/ner/logs/%x-%j.out
set -uo pipefail
D="$HOME/renascor-atlas-data/ner"
export E="$D/v2/editor_bench" JD="$D/v2/editor_bench/jobs/$SLURM_JOB_ID"
mkdir -p "$E/pred" "$E/logs" "$JD/claims"
echo "[eb-gpu] $(date +%T) job $SLURM_JOB_ID nodes $SLURM_JOB_NODELIST tasks $SLURM_NTASKS"
srun --kill-on-bad-exit=0 bash -c '
set -uo pipefail
s=$(printf "%02d" "$SLURM_PROCID"); LOG="$JD/task$s.log"
export CUDA_VISIBLE_DEVICES=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d, -f$((SLURM_LOCALID + 1)))
source "$HOME/renascor-atlas-data/ner/venv_train/bin/activate"; cd "$HOME/renascor-atlas"
export PYTHONPATH="$HOME/renascor-atlas/src" HF_HUB_CACHE="$HOME/renascor-atlas-data/ner/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
echo "[task $s] $(date "+%F %T") $(hostname) gpu $CUDA_VISIBLE_DEVICES $(nvidia-smi --query-gpu=name --format=csv,noheader -i "$CUDA_VISIBLE_DEVICES")" >> "$LOG"
for M in v2_xlmrxl_Sfull v2_xlmrxl_Sbal v2_eb2b_Sbal v2_eb2b_Sfull v2_eb610_Sbal v2_eb610_Sfull v2_xlmrl_Sbal v2_xlmrl_Sfull \
         v2_mdeberta_Sfull v2_mdeberta_Sbal r3_mdeberta_HM_gpu; do
  mkdir "$JD/claims/$M" 2>/dev/null || continue
  case "$M" in r3_mdeberta_HM_gpu) MD="$HOME/renascor-atlas-data/ner/models/r3_mdeberta_HM" ;; *) MD="$HOME/renascor-atlas-data/ner/train_v2/models/$M" ;; esac
  OUT="$E/pred/$M.jsonl.gz"; [ -s "$OUT" ] && continue
  t0=$(date +%s)
  python scripts/ner_audit_predict.py --model-dir "$MD" --data "$E/bench.jsonl" --out "$OUT.part" --words-from tokens > "$E/logs/$M.gpu.log" 2>&1 \
    && mv "$OUT.part" "$OUT"; rc=$?
  echo "[task $s] $(date "+%F %T") $M rc=$rc seconds=$(( $(date +%s) - t0 )) lines=$(zcat "$OUT" 2>/dev/null | wc -l)" >> "$LOG"
done
echo "[task $s] $(date "+%F %T") done" >> "$LOG"
'
echo "[eb-gpu] $(date +%T) all tasks finished"
cat "$JD"/task*.log
