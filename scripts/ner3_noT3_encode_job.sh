#!/bin/bash
# RenNER variant without CELT and CIDTC: per-seed modern_silver copies, then tokenise MIX_SFULL_LC25_LV4_NOT3_S{1,2,3} (and the filtered dev) for seeds 1-3 into the shared cache, on CPU
# (debug partition). Same arguments as the lc25-lv4 encode job and the seed runs. Gold v2, gold v3, dev3 and the editor bench are never read.
#   sbatch -p debug -c 24 --mem=160G -t 01:30:00 -J r8-noT3-enc -o ~/renascor-atlas-data/ner/train_noT3/logs/%x-%j.out scripts/ner3_noT3_encode_job.sh
set -u
set -f
D="$HOME/renascor-atlas-data/ner"
source "$D/venv_train/bin/activate"
cd "$HOME/renascor-atlas"
source scripts/ner_sets.sh
source scripts/ner_mixes.sh
source scripts/ner_mixes_noT3.sh
export PYTHONPATH="$HOME/renascor-atlas/src" HF_HUB_CACHE="$D/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true OMP_NUM_THREADS=6 RAYON_NUM_THREADS=6 PYTHONUNBUFFERED=1
TV="$D/train_noT3"; mkdir -p "$TV/encode" "$TV/logs"
one() {   # mix seed
  python -m ratlas.ner.train --name "enc_$1_$2" --model microsoft/mdeberta-v3-base ${!1} --dev "$DEV_NOT3" --out "$TV/encode" \
      --cache "$ND/cache" --max-len 256 --seed "$2" --encode-only > "$TV/encode/$1_s$2.log" 2>&1
  echo "[encode $1 seed $2] exit $?"
}
pre() {   # seed: the modern_silver sample of the lc25-lv4 run of that seed, minus the filtered records
  [ -s "$T3D/modern_silver_s$1_noT3.jsonl" ] && return 0
  python scripts/ner3_noT3_modern_presample.py --src "$AGG/modern_silver/train.jsonl" --cap 6000000 --seed "$1" --drop "$T3D/dropped.jsonl" \
      --out "$T3D/modern_silver_s$1_noT3.jsonl.part" --report "$T3D/modern_silver_s$1_noT3.json" && mv "$T3D/modern_silver_s$1_noT3.jsonl.part" "$T3D/modern_silver_s$1_noT3.jsonl"
  echo "[presample seed $1] exit $?"
}
pre 1 & pre 2 & pre 3 & wait
one MIX_SFULL_LC25_LV4_NOT3_S1 1 & one MIX_SFULL_LC25_LV4_NOT3_S2 2 & one MIX_SFULL_LC25_LV4_NOT3_S3 3 & wait
set +f
grep -h -E "train encoded|train windows|encode only|exit" "$TV"/encode/*.log
echo "all done $(date +%T)"
