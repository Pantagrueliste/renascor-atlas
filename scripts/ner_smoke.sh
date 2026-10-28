#!/usr/bin/env bash
# Run inside a GPU job: a 2000-record, 1-epoch run of every encoder, to catch tokenizer / model incompatibilities before the real runs.
set -uo pipefail
set -f
D="$HOME/renascor-atlas-data/ner"
source "$D/venv_train/bin/activate"
export PYTHONPATH="$HOME/renascor-atlas/src" HF_HUB_CACHE="$D/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=true PYTHONUNBUFFERED=1
O="$D/data_v1"
for spec in "smoke_hmbert64k dbmdz/bert-base-historic-multilingual-64k-td-cased" "smoke_mmbert jhu-clsp/mmBERT-base" "smoke_mdeberta microsoft/mdeberta-v3-base" \
            "smoke_eurobert210 $D/local_models/EuroBERT-210m --trust-remote-code" "smoke_hmteams hmteams/teams-base-historic-multilingual-discriminator"; do
  set -- $spec
  name=$1; model=$2; shift 2
  echo "=== $name"
  python -m ratlas.ner.train --name $name --model $model "$@" --train ours=$O/train.jsonl --limit-train 2000 --dev $O/dev.jsonl --test test=$O/test.jsonl \
     --epochs 1 --eval-every 1 --out "$D/runs" --models "$D/models_smoke" --cache "$D/cache" 2>&1 | grep -v "^Token indices" | tail -12
done
