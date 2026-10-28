#!/bin/bash
# Off-the-shelf models on the held-out test editions, predictions saved (same labels / windows / thresholds as the previous evaluation).
# MODE=train: venv_train (GLiNER-X large with the dev-selected threshold 0.7, Davlan XLM-R large, impresso mmBERT); MODE=flair: venv_flair.
set -u
A="$ND/audit"; P="$A/pred"; mkdir -p "$P"
T="$ND/data_v1/test.jsonl"
if [ "${MODE:-train}" = "train" ]; then
  python scripts/ner_audit_zs.py --backend gliner --model "$ND/local_models/gliner-x-large" --labels LOC=place,PER=person --threshold 0.7 --data "$T" --out "$P/zs_gliner_x_large__test.jsonl.gz"
  python scripts/ner_audit_zs.py --backend hf --model Davlan/xlm-roberta-large-ner-hrl --map loc=LOC,per=PER --data "$T" --out "$P/zs_davlan_xlmr_large__test.jsonl.gz"
  python scripts/ner_audit_zs.py --backend hf --model impresso-project/ner-hipe2020-mmbert-base-final --map loc=LOC,pers=PER --data "$T" --out "$P/zs_impresso_mmbert__test.jsonl.gz"
else
  python scripts/ner_audit_zs.py --backend flair --model flair/ner-english-large --map loc=LOC,per=PER --data "$T" --out "$P/zs_flair_ner_en_large__test.jsonl.gz"
  python scripts/ner_audit_zs.py --backend flair --model flair/ner-german-large --map loc=LOC,per=PER --data "$T" --out "$P/zs_flair_ner_de_large__test.jsonl.gz"
fi
