#!/bin/bash
# After the fix of the special tokens of tokenizers rebuilt from tokenizer.json (mmBERT): mmBERT again on test and seed; original pipeline eval-only;
# off-the-shelf token classifiers and GLiNER on the test editions with saved predictions.
set -u
A="$ND/audit"; T="$ND/data_v1/test.jsonl"
python scripts/ner_audit_predict.py --model-dir "$ND/models/r3_mmbert_base_HM" --data "$T" --out "$A/pred/r3_mmbert_base_HM__test.jsonl.gz"
python scripts/ner_audit_predict.py --model-dir "$ND/models/r3_mmbert_base_HM" --data "$A/seed/seed_passages.jsonl" --out "$A/pred_seed/r3_mmbert_base_HM.jsonl.gz"
python scripts/ner_audit_predict.py --model-dir "$ND/models/r3_mmbert_base_HM" --data "$ND/data_v1/dev.jsonl" --out "$A/pred/r3_mmbert_base_HM__dev.jsonl.gz"
export MODE=train
bash scripts/ner_audit_run_zs_test.sh
bash scripts/ner_audit_run_orig.sh
