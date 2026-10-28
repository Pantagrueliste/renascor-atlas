#!/bin/bash
# Re-evaluation of the best models on the held-out test editions: (1) the original pipeline in eval-only mode, separate output directory;
# (2) the independent inference (scripts/ner_audit_predict.py), predictions with span confidences.
set -u
A="$ND/audit"; mkdir -p "$A/pred" "$A/runs"
T="$ND/data_v1/test.jsonl"
for M in r3_mdeberta_HM r3_mmbert_base_HM r3_xlmr_large_H r3_hmteams_HM ft_xlmr_large_ours ft_hmteams_ours; do
  echo "=== $M original pipeline"
  python -m ratlas.ner.train --eval_only --name "$M" --load "$ND/models/$M" --test test=$T --save-preds test --out "$A/runs" --cache "$A/cache_$M" --tag _audit
  echo "=== $M independent inference"
  python scripts/ner_audit_predict.py --model-dir "$ND/models/$M" --data "$T" --out "$A/pred/${M}__test.jsonl.gz"
done
for M in r3_mdeberta_HM r3_mmbert_base_HM r3_xlmr_large_H; do
  python scripts/ner_audit_predict.py --model-dir "$ND/models/$M" --data "$ND/data_v1/dev.jsonl" --out "$A/pred/${M}__dev.jsonl.gz"
done
