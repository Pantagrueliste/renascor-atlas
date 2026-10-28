#!/bin/bash
# RenNER seeds 2 and 3: tokenise the training mixes of the seed runs into the shared cache, on CPU (debug partition). The cache key holds the
# seed (modern_silver is sampled with it), so every (mix, seed) is encoded once here, with the exact arguments of the training task.
# Gold v2, gold v3, dev3 and the editor bench are never read.
#   sbatch -p debug -c 24 --mem=160G -t 01:00:00 -J r6-encode -o ~/renascor-atlas-data/ner/train_seeds/logs/%x-%j.out scripts/ner3_seeds_encode_job.sh
set -u
set -f
D="$HOME/renascor-atlas-data/ner"
source "$D/venv_train/bin/activate"
cd "$HOME/renascor-atlas"
source scripts/ner_sets.sh
source scripts/ner_mixes.sh
export PYTHONPATH="$HOME/renascor-atlas/src" HF_HUB_CACHE="$D/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true OMP_NUM_THREADS=4 RAYON_NUM_THREADS=4 PYTHONUNBUFFERED=1
TV="$D/train_seeds"; mkdir -p "$TV/encode"
one() {   # mix seed
  python -m ratlas.ner.train --name "enc_$1_$2" --model microsoft/mdeberta-v3-base ${!1} --dev "$OURS/dev.jsonl" --out "$TV/encode" \
      --cache "$ND/cache" --max-len 256 --seed "$2" --encode-only > "$TV/encode/$1_s$2.log" 2>&1
  echo "[encode $1 seed $2] exit $?"
}
for s in 2 3; do one MIX_SFULL_LC25 $s & one MIX_SFULL $s & one MIX_HM $s & wait; done
set +f
grep -h -E "train encoded|train windows|encode only|exit" "$TV"/encode/*.log
echo "all done $(date +%T)"
