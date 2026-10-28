#!/bin/bash
# NER v2, large encoders, third smoke job (one A100): XLM-R XL with a LayerNorm before the heads (--head-norm): its last hidden states have a std of ~0.04.
# Two learning rates, activation checkpointing, the whole batch at once, 300 steps, full v1 dev at the end. Gold v2 is never read.
set -u
source scripts/ner_sets.sh
source scripts/ner_mixes.sh
S="$ND/v3/enc_smoke"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="$MIX_HM --dev $OURS/dev.jsonl --out $S/runs --models $S/models --cache $ND/cache --batch 32 --max-tokens 12000 --max-len 256 --seed 1 --warmup 0.06 --max-steps 300 --limit-dev 1000 --save-preds none"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for spec in "xlxl_hn_lr1e5|facebook/xlm-roberta-xl|1e-5|--grad-ckpt --head-norm" "xlxl_hn_lr2e5|facebook/xlm-roberta-xl|2e-5|--grad-ckpt --head-norm"; do
  IFS='|' read -r n MP LR EX <<< "$spec"
  echo "================ $n $MP lr $LR $(date +%T)"
  python -m ratlas.ner.train --name "smoke3_$n" --model "$MP" --lr "$LR" $EX --grad-accum 1 $COMMON
  echo "[smoke3 $n] exit $?"
  rm -f "$S/models/smoke3_$n/model.pt"
done
echo "all done $(date +%T)"
