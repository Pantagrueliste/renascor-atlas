#!/bin/bash
# NER v2, large encoders, fourth smoke job (one A100): EuroBERT 2.1B and 610m with the LayerNorm before the heads (--head-norm), as XLM-R XL needs it.
# The fastest setting of each model (no activation checkpointing; 2.1B in 2 micro-batches), 300 steps, full v1 dev at the end. Gold v2 is never read.
set -u
source scripts/ner_sets.sh
source scripts/ner_mixes.sh
S="$ND/v3/enc_smoke"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="$MIX_HM --dev $OURS/dev.jsonl --out $S/runs --models $S/models --cache $ND/cache --batch 32 --max-tokens 12000 --max-len 256 --seed 1 --warmup 0.06 --max-steps 300 --limit-dev 1000 --save-preds none"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for spec in "eb2b_hn|$ND/local_models/EuroBERT-2.1B|2e-5|--trust-remote-code --force-bos --head-norm|2" "eb610_hn|$ND/local_models/EuroBERT-610m|2.5e-5|--trust-remote-code --force-bos --head-norm|1"; do
  IFS="|" read -r n MP LR EX ACC <<< "$spec"
  echo "================ $n $MP lr $LR $(date +%T)"
  python -m ratlas.ner.train --name "smoke4_$n" --model "$MP" --lr "$LR" $EX --grad-accum "$ACC" $COMMON
  echo "[smoke4 $n] exit $?"
  rm -f "$S/models/smoke4_$n/model.pt"
done
echo "all done $(date +%T)"
