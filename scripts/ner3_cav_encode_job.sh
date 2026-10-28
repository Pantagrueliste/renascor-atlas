#!/bin/bash
# Cavriana transfer test: tokenise the two mixes without Cavriana and the v1 dev without Cavriana into the shared cache (CPU, debug).
#   sbatch -p debug -c 16 --mem=120G -t 01:30:00 -J cav-encode scripts/ner3_cav_encode_job.sh
#SBATCH -o /home/cag30/renascor-atlas-data/ner/cav_transfer/logs/%x-%j.out
set -u; set -f
D="$HOME/renascor-atlas-data/ner"
source "$D/venv_train/bin/activate"; cd "$HOME/renascor-atlas"
source scripts/ner_sets.sh; source scripts/ner_mixes.sh; source scripts/ner3_cav_mixes.sh
export PYTHONPATH="$HOME/renascor-atlas/src" HF_HUB_CACHE="$D/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true OMP_NUM_THREADS=8 RAYON_NUM_THREADS=8 PYTHONUNBUFFERED=1
TV="$D/cav_transfer/train"; mkdir -p "$TV/encode"
echo "mixes: SFULL_NOCAV=$MIX_SFULL_NOCAV"; echo "HM_NOCAV=$MIX_HM_NOCAV"; echo "dev=$CAV_DEV"
for mixv in MIX_SFULL_NOCAV MIX_HM_NOCAV; do
  python -m ratlas.ner.train --name "enc_$mixv" --model microsoft/mdeberta-v3-base ${!mixv} --dev "$CAV_DEV" --out "$TV/encode" \
      --cache "$ND/cache" --max-len 256 --seed 1 --encode-only > "$TV/encode/$mixv.log" 2>&1 &
done
wait
grep -h -E "train encoded|train windows|encode only|Error|error" "$TV"/encode/*.log
echo "done $(date +%T)"
