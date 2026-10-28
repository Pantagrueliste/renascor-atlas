#!/bin/bash
# Seed set (30 passages of untagged texts): predictions of the models that are compared in the report. SEED = path of the passages.
# MODE=train (venv_train: our fine-tuned models + GLiNER) or MODE=flair (venv_flair: flair taggers).
set -u
A="$ND/audit"; P="$A/pred_seed"; mkdir -p "$P"
SEED="${SEED:-$A/seed/seed_passages.jsonl}"
if [ "${MODE:-train}" = "train" ]; then
  for M in ft_xlmr_large_ours ft_hmteams_ours r3_mdeberta_HM r3_xlmr_large_H r3_mmbert_base_HM; do
    python scripts/ner_audit_predict.py --model-dir "$ND/models/$M" --data "$SEED" --out "$P/${M}.jsonl.gz" --words-from tokens
  done
  python scripts/ner_audit_zs.py --backend gliner --model "$ND/local_models/gliner-x-large" --labels LOC=place,PER=person --threshold 0.7 --data "$SEED" --out "$P/zs_gliner_x_large.jsonl.gz"
  python scripts/ner_audit_zs.py --backend hf --model Davlan/xlm-roberta-large-ner-hrl --map loc=LOC,per=PER --data "$SEED" --out "$P/zs_davlan_xlmr_large.jsonl.gz"
else
  python scripts/ner_audit_zs.py --backend flair --model flair/ner-english-large --map loc=LOC,per=PER --data "$SEED" --out "$P/zs_flair_ner_en_large.jsonl.gz"
  python scripts/ner_audit_zs.py --backend flair --model flair/ner-german-large --map loc=LOC,per=PER --data "$SEED" --out "$P/zs_flair_ner_de_large.jsonl.gz"
fi
