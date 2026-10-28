#!/usr/bin/env bash
# CPU job (debug partition): score models on the 360-passage LLM sample so that the LLM and the other systems are compared on exactly the same passages.
#   sbatch --partition=debug -c 16 --mem=48G -t 120 --wrap "bash ~/renascor-atlas/scripts/ner_eval_llm_sample.sh ft:ft_xlmr_base_ours hf:zs_davlan_xlmr_large:Davlan/xlm-roberta-large-ner-hrl:loc=LOC,per=PER ..."
# ft:RUN -> fine-tuned run (models/RUN) ; hf:NAME:MODEL:MAP -> token-classification model ; gl:NAME:MODEL:LABELS:THRESHOLD -> GLiNER (threshold selected on dev in the full run)
set -uo pipefail
set -f
D="$HOME/renascor-atlas-data/ner"
source "$D/venv_train/bin/activate"
export PYTHONPATH="$HOME/renascor-atlas/src" HF_HUB_CACHE="$D/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=true PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=
SAMPLE="${SAMPLE:-llm_sample}"      # file name (without .jsonl) in $D/samples
SUF="${SUF:-llm}"                    # suffix of the zero-shot run names and of the fine-tuned metrics file
S="llm_sample=$D/samples/$SAMPLE.jsonl"
for spec in "$@"; do
  IFS=: read -r kind name model map thr <<< "$spec"
  case "$kind" in
    ft) python -m ratlas.ner.train --eval-only --name "$name" --load "$D/models/$name" --out "$D/runs" --models "$D/models" --cache "$D/cache" --eval "$S" --tag "_$SUF" --save-preds none ;;
    hf) python -m ratlas.ner.zeroshot --backend hf --name "${name}_$SUF" --model "$model" --map "$map" --set "$S" --out "$D/runs" ;;
    gl) python -m ratlas.ner.zeroshot --backend gliner --name "${name}_$SUF" --model "$model" --labels "$map" --thresholds "$thr" --set "$S" --out "$D/runs" ;;
  esac
done
