#!/bin/bash
# Editor-tag bench (Hypatia debug, CPU): encoder predictions with the audit predictor (scripts/ner_audit_predict.py), the bench cut in
# two halves so that one unit (model x half) fits easily in the 2 h limit. Finished units are skipped, so the same job can be resubmitted
# (or chained with --dependency=afterany) until every model is done. A unit is written to a .part file, then renamed.
#   sbatch -p debug -c 24 --mem=64G -t 02:00:00 -J eb-enc scripts/ner_editor_bench_job_encoders.sh [MODEL_DIR_OR_NAME ...]
# Default models: the four v1 encoders. An argument is a run name under $D/models or a full model directory (config.json, model.pt,
# tokenizer/). Output: $E/pred/<name>.jsonl.gz (one line per bench passage).
#SBATCH -o /home/cag30/renascor-atlas-data/ner/logs/%x-%j.out
set -uo pipefail
D="$HOME/renascor-atlas-data/ner"; E="${EB_DIR:-$D/v2/editor_bench}"; P="$E/pred"; mkdir -p "$P/parts"
source "$D/venv_train/bin/activate"
export PYTHONPATH="$HOME/renascor-atlas/src" HF_HUB_CACHE="$D/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} PYTHONUNBUFFERED=1
cd "$HOME/renascor-atlas"
echo "[eb-enc] $(date +%T) $(hostname) job ${SLURM_JOB_ID:-} cpus=${SLURM_CPUS_PER_TASK:-?}"
if [ ! -s "$E/half_1.jsonl" ]; then
  python - "$E/bench.jsonl" "$E" <<'PY'
import sys
lines = [l for l in open(sys.argv[1], encoding='utf-8') if l.strip()]
h = (len(lines) + 1) // 2
for k, part in enumerate((lines[:h], lines[h:])):
    open(f'{sys.argv[2]}/half_{k}.jsonl.tmp', 'w', encoding='utf-8').writelines(part)
PY
  mv "$E/half_0.jsonl.tmp" "$E/half_0.jsonl"; mv "$E/half_1.jsonl.tmp" "$E/half_1.jsonl"
fi
MODELS=("$@"); [ ${#MODELS[@]} -eq 0 ] && MODELS=(r3_mdeberta_HM r3_mmbert_base_HM r3_hmteams_HM r3_xlmr_large_H)
rc=0
for M in "${MODELS[@]}"; do
  if [ -d "$M" ]; then MD="$M"; NAME="$(basename "$M")"; else MD="$D/models/$M"; NAME="$M"; fi
  [ -s "$P/$NAME.jsonl.gz" ] && { echo "skip $NAME"; continue; }
  for k in 0 1; do
    U="$P/parts/${NAME}__half_$k.jsonl.gz"
    [ -s "$U" ] && { echo "skip $NAME half $k"; continue; }
    t0=$(date +%s); echo "=== $NAME half $k $(date +%T)"
    python scripts/ner_audit_predict.py --model-dir "$MD" --data "$E/half_$k.jsonl" --out "$U.part" --words-from tokens \
      && mv "$U.part" "$U" || { rc=1; echo "=== FAILED $NAME half $k"; }
    echo "=== $NAME half $k seconds=$(( $(date +%s) - t0 ))"
  done
  if [ -s "$P/parts/${NAME}__half_0.jsonl.gz" ] && [ -s "$P/parts/${NAME}__half_1.jsonl.gz" ]; then
    cat "$P/parts/${NAME}__half_0.jsonl.gz" "$P/parts/${NAME}__half_1.jsonl.gz" > "$P/$NAME.jsonl.gz.part" && mv "$P/$NAME.jsonl.gz.part" "$P/$NAME.jsonl.gz"
    echo "=== $NAME complete: $(zcat "$P/$NAME.jsonl.gz" | wc -l) lines"
  fi
done
echo "[eb-enc] exit $rc $(date +%T)"
exit $rc
