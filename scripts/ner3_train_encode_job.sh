#!/bin/bash
# NER v2 (RenNER) training grid: tokenise the S-bal and S-full mixes, the v1 dev and the final evaluation sets into the shared cache, on CPU
# (debug partition), one process per tokenizer. Gold v2 is never read.
#   sbatch -p debug -c 24 --mem=160G -t 01:30:00 -J r4-encode -o ~/renascor-atlas-data/ner/train_v2/logs/%x-%j.out scripts/ner3_train_encode_job.sh
set -u
set -f
D="$HOME/renascor-atlas-data/ner"
source "$D/venv_train/bin/activate"
cd "$HOME/renascor-atlas"
source scripts/ner_sets.sh
source scripts/ner_mixes.sh
export PYTHONPATH="$HOME/renascor-atlas/src" HF_HUB_CACHE="$D/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true OMP_NUM_THREADS=6 RAYON_NUM_THREADS=6 PYTHONUNBUFFERED=1
TV="$D/train_v2"; mkdir -p "$TV/encode"
EVALS="--set test=$OURS/test.jsonl --set ext_hipe=$EXT/hipe2022__*.jsonl --set ext_hist=$EXT/europeana__*.jsonl;$EXT/reichsanzeiger__*.jsonl;$EXT/geoedda__*.jsonl;$EXT/globalise__*.jsonl;$EXT/earlymodern_german__*.jsonl;$EXT/sdhk_old_swedish__*.jsonl;$EXT/kind__*.jsonl;$EXT/latin_*.jsonl --set ext_charters=$EXT/medieval_charters__*.jsonl --set dev=$OURS/dev.jsonl"
one() {   # tag model "extra" mixes...
  local tag="$1" model="$2" extra="$3"; shift 3
  for mixv in "$@"; do
    python -m ratlas.ner.train --name "enc_$tag" --model "$model" $extra ${!mixv} --dev "$OURS/dev.jsonl" --out "$TV/encode" \
        --cache "$ND/cache" --max-len 256 --seed 1 --encode-only
    echo "[encode $tag $mixv] exit $?"
  done
  python scripts/ner3_train_encode_eval.py --model "$model" $extra --cache "$ND/cache" $EVALS
  echo "[encode-eval $tag] exit $?"
}
one mdeberta microsoft/mdeberta-v3-base "" MIX_SBAL MIX_SFULL > "$TV/encode/mdeberta.log" 2>&1 &
one xlmrl FacebookAI/xlm-roberta-large "" MIX_SBAL > "$TV/encode/xlmrl.log" 2>&1 &
one xlmrxl facebook/xlm-roberta-xl "" MIX_SBAL MIX_SFULL > "$TV/encode/xlmrxl.log" 2>&1 &
one eurobert "$ND/local_models/EuroBERT-2.1B" "--trust-remote-code" MIX_SBAL MIX_SFULL > "$TV/encode/eurobert.log" 2>&1 &
wait
# EuroBERT-610m has the same tokenizer as 2.1B (same tokenizer.json): its cache keys are the same, this only checks it
one eurobert610 "$ND/local_models/EuroBERT-610m" "--trust-remote-code" MIX_SBAL > "$TV/encode/eurobert610.log" 2>&1
set +f
grep -h -E "train encoded|train windows|exit|encode-eval\]" "$TV"/encode/*.log
echo "all done $(date +%T)"
