#!/bin/bash
# NER layer v3 (TEI editions, repaired extra documents): one GPU job (<= 4 h), SET=tei or SET=extra.
# Non-Spanish: r3_xlmr_large_H main, the other three encoders on the passages with a span of confidence >= 0.9 (same as layer v1, ner2_layer_gpu.sbatch).
# Spanish: r3_mdeberta_HM main, the other encoders on every passage with a span (ner2_layer_es_gpu.sbatch) -> <set>/es_mdeberta.
# The language is part of every chunk name (--tag v3: v3__<lang>__<shard>__cNNN), so two shards with one name in two languages cannot collide.
# Resumable by chunk. Needs <set>/plan/worklist.txt and worklist_es.txt (made by ner2_layer_v3_plan.py).
#   SET=tei sbatch --job-name=ner_v3_tei slurm/ner2_layer_v3_gpu.sbatch
#SBATCH --partition=gpu.A100,gpu.L40S,gpu.A30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cag30/renascor-atlas-data/ner/layer_v3/logs/%x-%j.out
set -uo pipefail
D="$HOME/renascor-atlas-data/ner"
SET="${SET:?SET=tei|extra|seen}"
O="$D/layer_v3/$SET"
case "$SET" in
  tei)   TR="$D/layer_tei";               DOCS="$D/layer_tei/docs.parquet" ;;
  seen)  TR="$D/layer_tei/seen";          DOCS="$D/layer_tei/seen/docs.parquet" ;;
  extra) TR="$D/layer_extra/extract";     DOCS="$D/layer_extra/extract/docs_read.parquet" ;;
  *) echo "unknown SET" >&2; exit 2 ;;
esac
source "$D/venv_train/bin/activate"
export PYTHONPATH="$HOME/renascor-atlas/src" HF_HUB_CACHE="$D/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
cd "$HOME/renascor-atlas"
echo "[ner2-v3-gpu $SET] $(date +%T) $(hostname) job ${SLURM_JOB_ID:-} gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
python scripts/ner2_layer_run.py --texts-root "$TR" --shards-file "$O/plan/worklist.txt" --tag v3 --docs "$DOCS" --out "$O" \
  --models-dir "$D/models" --stop-after "${STOP_AFTER:-6600}"
rc=$?
if [ $rc -eq 0 ]; then
  mkdir -p "$O/es_mdeberta/logs"
  python scripts/ner2_layer_run_es.py --texts-root "$TR" --shards-file "$O/plan/worklist_es.txt" --tag v3 --docs "$DOCS" --out "$O/es_mdeberta" \
    --models-dir "$D/models" --select-conf 0.0 --stop-after "${STOP_AFTER_ES:-3000}"
  rc=$?
fi
echo "[ner2-v3-gpu $SET] exit $rc $(date +%T)"
exit $rc
