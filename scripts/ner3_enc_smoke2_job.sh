#!/bin/bash
# NER v2, large encoders, second smoke job (one A100): XLM-R XL predicted no entity after 300 steps at lr 1e-5 (loss at the all-O level).
# Reference: XLM-R large with the same 300-step schedule (the v1 recipe of r3_xlmr_large: lr 2e-5); then XL at lr 2e-5 and 3e-5 with activation
# checkpointing and the whole batch at once (probe A of the first job had failed). Full v1 dev at the end of each run. Gold v2 is never read.
set -u
source scripts/ner_sets.sh
source scripts/ner_mixes.sh
S="$ND/v3/enc_smoke"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="$MIX_HM --dev $OURS/dev.jsonl --out $S/runs --models $S/models --cache $ND/cache --batch 32 --max-tokens 12000 --max-len 256 --seed 1 --warmup 0.06 --max-steps 300 --limit-dev 1000 --save-preds none"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for spec in "ref_xlmr_large|FacebookAI/xlm-roberta-large|2e-5|" "xlxl_lr2e5|facebook/xlm-roberta-xl|2e-5|--grad-ckpt" "xlxl_lr3e5|facebook/xlm-roberta-xl|3e-5|--grad-ckpt"; do
  IFS='|' read -r n MP LR EX <<< "$spec"
  echo "================ $n $MP lr $LR $(date +%T)"
  python -m ratlas.ner.train --name "smoke2_$n" --model "$MP" --lr "$LR" $EX --grad-accum 1 $COMMON
  echo "[smoke2 $n] exit $?"
  rm -f "$S/models/smoke2_$n/model.pt"
done
echo "all done $(date +%T)"
