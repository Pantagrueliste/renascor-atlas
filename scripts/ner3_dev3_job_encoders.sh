#!/bin/bash
# dev3 job (Hypatia debug, CPU): the four R3 encoders on dev3 (LOC and PER), same predictor as the v1 audit and gold v2 (ner_audit_predict.py).
#   sbatch -p debug -c 16 --mem=48G -t 02:00:00 -J r3h-enc scripts/ner3_dev3_job_encoders.sh
set -uo pipefail
D="$HOME/renascor-atlas-data/ner"; W="$D/v3/dev3"; P="$W/pred_enc"; mkdir -p "$P"
source "$D/venv_train/bin/activate"
export PYTHONPATH="$HOME/renascor-atlas/src" HF_HUB_CACHE="$D/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} PYTHONUNBUFFERED=1
cd "$HOME/renascor-atlas"
rc=0
for M in r3_mdeberta_HM r3_xlmr_large_H r3_mmbert_base_HM r3_hmteams_HM; do
  [ -s "$P/$M.jsonl.gz" ] && { echo "skip $M"; continue; }
  t0=$(date +%s); echo "=== $M $(date +%T)"
  python scripts/ner_audit_predict.py --model-dir "$D/models/$M" --data "$W/passages.jsonl" --out "$P/$M.jsonl.gz" --words-from tokens || rc=1
  echo "=== $M seconds=$(( $(date +%s) - t0 ))"
done
exit $rc
