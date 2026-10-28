#!/bin/bash
# RenNER Latin v4: lowercase copies (25 %, seed 21, as lc25) from the S-full sets with the v4 Latin silver, then tokenise MIX_SFULL_LC25_LV4
# for seeds 1-3 into the shared cache, on CPU (debug partition). Same arguments as the v2.1 encode job and the seed runs.
# Gold v2, gold v3, dev3 and the editor bench are never read.
#   sbatch -p debug -c 24 --mem=160G -t 01:30:00 -J r6-lv4-enc -o ~/renascor-atlas-data/ner/train_lv4/logs/%x-%j.out scripts/ner3_lv4_encode_job.sh
set -u
set -f
D="$HOME/renascor-atlas-data/ner"
source "$D/venv_train/bin/activate"
cd "$HOME/renascor-atlas"
source scripts/ner_sets.sh
source scripts/ner_mixes.sh
source scripts/ner_mixes_lv4.sh
export PYTHONPATH="$HOME/renascor-atlas/src" HF_HUB_CACHE="$D/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true OMP_NUM_THREADS=6 RAYON_NUM_THREADS=6 PYTHONUNBUFFERED=1
TV="$D/train_lv4"; mkdir -p "$TV/encode" "$TV/logs"
A="$V21/aug/aug_lc25_lv4.jsonl"
if [ ! -s "$A" ]; then
  SRC="--src ours=$OURS/train.jsonl --src hist_gold=$AGG/hist_gold/train.jsonl --src silver_bal_es=$SIL/silver_balanced_es.jsonl --src silver_bal_it=$SIL/silver_balanced_it.jsonl --src silver_bal_la=$SIL_LV4/silver_balanced_la.jsonl --src silver_res_es=$SIL/silver_reserve_es.jsonl --src silver_res_it=$SIL/silver_reserve_it.jsonl"
  python scripts/ner3_v21_lc_augment.py $SRC --sfull-passages 627600 --share 0.25 --p-all 0.3 --p-keep-span 0.2 --seed 21 \
      --out "$A" --report "$V21/aug/aug_lc25_lv4.json" > "$V21/aug/aug_lc25_lv4.log" 2>&1
  echo "[aug lc25_lv4] exit $?"
fi
one() {   # mix seed
  python -m ratlas.ner.train --name "enc_$1_$2" --model microsoft/mdeberta-v3-base ${!1} --dev "$OURS/dev.jsonl" --out "$TV/encode" \
      --cache "$ND/cache" --max-len 256 --seed "$2" --encode-only > "$TV/encode/$1_s$2.log" 2>&1
  echo "[encode $1 seed $2] exit $?"
}
one MIX_SFULL_LC25_LV4 1 & one MIX_SFULL_LC25_LV4 2 & one MIX_SFULL_LC25_LV4 3 & wait
set +f
grep -h -E "train encoded|train windows|encode only|exit" "$TV"/encode/*.log
echo "all done $(date +%T)"
