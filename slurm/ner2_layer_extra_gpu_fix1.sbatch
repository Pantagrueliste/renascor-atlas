#!/bin/bash
# NER layer, extra documents: one GPU job (<= 4 h). Main run (XLM-R large main, other encoders on the passages with a span of confidence >= 0.9) on the shards of
# layer_extra/extract/plan/worklist.txt, then the Spanish run (mDeBERTa main, other encoders on every passage with a span) on worklist_es.txt into layer_extra/es_mdeberta.
# Resumable by chunk. Same models and settings as the layer v1 (ner2_layer_gpu.sbatch, ner2_layer_es_gpu.sbatch).
#   sbatch --job-name=ner_layer_extra slurm/ner2_layer_extra_gpu.sbatch
#SBATCH --partition=gpu.A100,gpu.L40S,gpu.A30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cag30/renascor-atlas-data/ner/layer_extra/logs/%x-%j.out
set -uo pipefail
D="$HOME/renascor-atlas-data/ner"
L="$D/layer_extra"
source "$D/venv_train/bin/activate"
export PYTHONPATH="$HOME/renascor-atlas/src" HF_HUB_CACHE="$D/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
cd "$HOME/renascor-atlas"
echo "[ner2-layer-extra-gpu] $(date +%T) $(hostname) job ${SLURM_JOB_ID:-} gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
python scripts/ner2_layer_run.py --texts-root "$L/extract" --shards-file "$L/extract/plan/worklist_fix1.txt" --tag fix1 --docs "$L/extract/docs_read.parquet" --out "$L" \
  --models-dir "$D/models" --stop-after "${STOP_AFTER:-6600}"
rc=$?
echo "[ner2-layer-extra-gpu] exit $rc $(date +%T)"
exit $rc
