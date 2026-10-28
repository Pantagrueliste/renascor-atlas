#!/bin/bash
# NER layer v1, one GPU job (<= 4 h): scripts/ner2_layer_run.py on the shards of SHARDS_FILE (one path per line, relative to the texts root); resumable by chunk.
#   scripts/ner2_layer_submit.sh ner_layer_a        (checks the shared GPU rule first)
#SBATCH --partition=gpu.A100,gpu.L40S,gpu.A30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cag30/renascor-atlas-data/ner/layer_v1/logs/%x-%j.out
set -uo pipefail
D="$HOME/renascor-atlas-data/ner"
L="$D/layer_v1"
source "$D/venv_train/bin/activate"
export PYTHONPATH="$HOME/renascor-atlas/src" HF_HUB_CACHE="$D/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
cd "$HOME/renascor-atlas"
echo "[ner2-layer-gpu] $(date +%T) $(hostname) job ${SLURM_JOB_ID:-} gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
# stop cleanly 12 minutes before the time limit (a chunk takes about 10 minutes)
python scripts/ner2_layer_run.py --texts-root "$L/extract" --shards-file "$SHARDS_FILE" --docs "$L/extract/docs_read.parquet" --out "$L" --models-dir "$D/models" \
  --stop-after "${STOP_AFTER:-13200}" ${EXTRA:-}
rc=$?
echo "[ner2-layer-gpu] exit $rc $(date +%T)"
exit $rc
