#!/bin/bash
# NER layer, Spanish rerun with mDeBERTa as the main model (one GPU job). Writes only under layer_v1/es_mdeberta/.
#SBATCH --job-name=ner_es_mdeberta
#SBATCH --partition=gpu.A100,gpu.L40S,gpu.A30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cag30/renascor-atlas-data/ner/layer_v1/es_mdeberta/logs/%x-%j.out
set -uo pipefail
D="$HOME/renascor-atlas-data/ner"
L="$D/layer_v1"
O="$L/es_mdeberta"
source "$D/venv_train/bin/activate"
export PYTHONPATH="$HOME/renascor-atlas/src" HF_HUB_CACHE="$D/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
cd "$HOME/renascor-atlas"
echo "[ner-es] $(date +%T) $(hostname) job ${SLURM_JOB_ID:-} gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
(cd "$L/extract/texts" && ls es/*.jsonl.gz) > "$O/shards_es.txt"
python scripts/ner2_layer_run_es.py --texts-root "$L/extract/texts" --shards-file "$O/shards_es.txt" --docs "$L/extract/docs_read.parquet" --out "$O" \
  --models-dir "$D/models" --doc-filter "$O/es_doc_ids.txt" --select-conf 0.0 --stop-after 6600
rc=$?
echo "[ner-es] exit $rc $(date +%T)"
exit $rc
