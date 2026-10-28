#!/bin/bash
# gold v3 job (Hypatia debug, CPU): the four R3 encoders on the 581 gold v3 passages (LOC and PER), same predictor as dev3 and gold v2
# (ner_audit_predict.py). Proposals for the owner's review only; never used to pick a model.
#   sbatch -p debug -c 16 --mem=48G -t 01:30:00 -J r3g3-enc -o $D/gold3/logs/%x-%j.out scripts/ner3_gold3_job_encoders.sh
# Other input / output (2026-10-24, merged option passages): G3_IN=$D/gold3/in/g3n_all.jsonl G3_PRED=$D/gold3/pred_enc_new sbatch ...
set -uo pipefail
D="$HOME/renascor-atlas-data/ner"; W="$D/gold3"; P="${G3_PRED:-$W/pred_enc}"; IN="${G3_IN:-$W/in/g3_all.jsonl}"; mkdir -p "$P"
source "$D/venv_train/bin/activate"
export PYTHONPATH="$HOME/renascor-atlas/src" HF_HUB_CACHE="$D/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} PYTHONUNBUFFERED=1
cd "$HOME/renascor-atlas"
rc=0
for M in r3_mdeberta_HM r3_xlmr_large_H r3_mmbert_base_HM r3_hmteams_HM; do
  [ -s "$P/$M.jsonl.gz" ] && { echo "skip $M"; continue; }
  t0=$(date +%s); echo "=== $M $(date +%T)"
  python scripts/ner_audit_predict.py --model-dir "$D/models/$M" --data "$IN" --out "$P/$M.jsonl.gz" --words-from tokens || rc=1
  echo "=== $M seconds=$(( $(date +%s) - t0 ))"
done
exit $rc
