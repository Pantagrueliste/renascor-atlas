#!/bin/bash
# RenNER v2, after the dev3 selection: CPU speed on dev3 (358 passages) and, last, predictions on the gold v2 passages, for the models given.
#   sbatch -p debug -c 24 --mem=96G -t 02:00:00 -J r4-cpu scripts/ner3_eval_v2_cpu_job.sh GOLD2_PASSAGES MODEL [MODEL ...]
# Gold v2 is only scored afterwards, on the Mac (scripts/ner3_eval_v2_gold2.py); this job only predicts on its passages.
#SBATCH -o /home/cag30/renascor-atlas-data/ner/train_v2/eval/logs/%x-%j.out
set -uo pipefail
D="$HOME/renascor-atlas-data/ner"; EV="$D/train_v2/eval"
G2="$1"; shift
mkdir -p "$EV/speed" "$EV/pred_gold2"
source "$D/venv_train/bin/activate"
cd "$HOME/renascor-atlas"
export PYTHONPATH="$HOME/renascor-atlas/src" HF_HUB_CACHE="$D/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24} CUDA_VISIBLE_DEVICES=""
echo "[r4c] $(date +%T) $(hostname) cpus ${SLURM_CPUS_PER_TASK:-?} $(lscpu | grep 'Model name' | sed 's/  */ /g')"
for M in "$@"; do
  case "$M" in r3_*) MD="$D/models/$M" ;; *) MD="$D/train_v2/models/$M" ;; esac
  t0=$(date +%s.%N)
  python scripts/ner_audit_predict.py --model-dir "$MD" --data "$D/v3/dev3/passages.jsonl" --out "$EV/speed/$M.cpu.jsonl.gz" --words-from tokens > "$EV/logs/$M.cpu.log" 2>&1
  echo "[r4c] $(date +%T) $M cpu rc=$? wall=$(echo "$(date +%s.%N) - $t0" | bc)"
done
for M in "$@"; do
  case "$M" in r3_*) MD="$D/models/$M" ;; *) MD="$D/train_v2/models/$M" ;; esac
  python scripts/ner_audit_predict.py --model-dir "$MD" --data "$G2" --out "$EV/pred_gold2/$M.jsonl.gz" --words-from tokens > "$EV/logs/$M.gold2.log" 2>&1
  echo "[r4c] $(date +%T) $M gold2 rc=$?"
done
