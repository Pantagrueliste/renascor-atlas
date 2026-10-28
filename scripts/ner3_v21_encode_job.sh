#!/bin/bash
# NER v2.1 (RenNER): build the lowercase-name copies (scripts/ner3_v21_lc_augment.py) and tokenise the S-full + copies mixes into the
# shared cache, on CPU (debug partition). Gold v2, gold v3, dev3 and the editor bench are never read.
#   sbatch -p debug -c 24 --mem=160G -t 01:30:00 -J r5-encode -o ~/renascor-atlas-data/ner/train_v21/logs/%x-%j.out scripts/ner3_v21_encode_job.sh
set -u
set -f
D="$HOME/renascor-atlas-data/ner"
source "$D/venv_train/bin/activate"
cd "$HOME/renascor-atlas"
source scripts/ner_sets.sh
source scripts/ner_mixes.sh
export PYTHONPATH="$HOME/renascor-atlas/src" HF_HUB_CACHE="$D/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true OMP_NUM_THREADS=6 RAYON_NUM_THREADS=6 PYTHONUNBUFFERED=1
TV="$D/train_v21"; mkdir -p "$TV/aug" "$TV/encode" "$TV/data"
# data_v1 train without the MAP passages (MAP: identifiers only, never in training)
if [ ! -s "$OURS_V21" ]; then
  grep -v '"edition":"map_of_early_modern_london"' "$OURS/train.jsonl" > "$OURS_V21.tmp" && mv "$OURS_V21.tmp" "$OURS_V21"
fi
echo "[nomap] $(wc -l < "$OURS/train.jsonl") -> $(wc -l < "$OURS_V21") passages; MAP left: $(grep -c -i map_of_early_modern "$OURS_V21")"
SRC="--src ours=$OURS/train.jsonl --src hist_gold=$AGG/hist_gold/train.jsonl --src silver_bal_es=$SIL/silver_balanced_es.jsonl --src silver_bal_it=$SIL/silver_balanced_it.jsonl --src silver_bal_la=$SIL/silver_balanced_la.jsonl --src silver_res_es=$SIL/silver_reserve_es.jsonl --src silver_res_it=$SIL/silver_reserve_it.jsonl"
aug() {  # tag share p_all
  [ -s "$TV/aug/aug_$1.jsonl" ] && return
  python scripts/ner3_v21_lc_augment.py $SRC --sfull-passages 627600 --share "$2" --p-all "$3" --p-keep-span 0.2 --seed 21 \
      --out "$TV/aug/aug_$1.jsonl" --report "$TV/aug/aug_$1.json" > "$TV/aug/aug_$1.log" 2>&1
  echo "[aug $1] exit $?"
}
aug lc12 0.12 0.3 &
aug lc25 0.25 0.3 &
aug lc12n 0.12 0.0 &
wait
one() {   # tag model mixes...
  local tag="$1" model="$2"; shift 2
  for mixv in "$@"; do
    python -m ratlas.ner.train --name "enc_$tag" --model "$model" ${!mixv} --dev "$OURS/dev.jsonl" --out "$TV/encode" \
        --cache "$ND/cache" --max-len 256 --seed 1 --encode-only
    echo "[encode $tag $mixv] exit $?"
  done
}
one mdeberta microsoft/mdeberta-v3-base MIX_SFULL_V21 MIX_SFULL_LC12 MIX_SFULL_LC25 MIX_SFULL_LC12N > "$TV/encode/mdeberta.log" 2>&1 &
one xlmrxl facebook/xlm-roberta-xl MIX_SFULL_V21 MIX_SFULL_LC12 MIX_SFULL_LC25 MIX_SFULL_LC12N > "$TV/encode/xlmrxl.log" 2>&1 &
wait
set +f
grep -h -E "train encoded|train windows|exit" "$TV"/encode/*.log
echo "all done $(date +%T)"
