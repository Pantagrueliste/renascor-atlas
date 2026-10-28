#!/bin/bash
# CPU check of the numpy inference against the v1 predictions (seed passages: 4 encoders; a sample of the test set: the main model).
set -u
D="$HOME/renascor-atlas-data/ner"; L="$D/layer_v1"
cd "$HOME/renascor-atlas"
for M in r3_xlmr_large_H r3_mdeberta_HM r3_mmbert_base_HM r3_hmteams_HM; do
  echo "=== seed $M"
  python scripts/ner2_layer_validate.py --model-dir "$D/models/$M" --data "$D/audit/seed/seed_passages.jsonl" --ref "$D/audit/pred_seed/$M.jsonl.gz"
done
echo "=== test sample r3_xlmr_large_H"
python scripts/ner2_layer_validate.py --model-dir "$D/models/r3_xlmr_large_H" --data "$L/validate/test_sample.jsonl" --ref "$D/audit/pred/r3_xlmr_large_H__test.jsonl.gz"
