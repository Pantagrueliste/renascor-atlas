#!/bin/bash
# Re-evaluation of the best models with the ORIGINAL pipeline (ratlas.ner.train in eval-only mode), separate output directory.
set -u
A="$ND/audit"; mkdir -p "$A/runs"
T="$ND/data_v1/test.jsonl"
for M in r3_mdeberta_HM r3_mmbert_base_HM r3_xlmr_large_H r3_hmteams_HM ft_xlmr_large_ours; do
  echo "=== $M original pipeline"
  python -m ratlas.ner.train --eval-only --name "$M" --load "$ND/models/$M" --test test=$T --save-preds test --out "$A/runs" --cache "$A/cache_$M" --tag _audit
done
