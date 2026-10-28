#!/bin/bash
# Reference set v2 (gold2): all predictions of one GPU job, in sequence. Needs ND (set by slurm/ner2_gpu.sbatch) and PASSAGES (passages.jsonl on the cluster).
#   encoders (4 R3 models)  -> venv_train, scripts/ner_audit_predict.py (span confidences = mean softmax probability of the span)
#   flair ner-english-large -> venv_flair, English passages only, scripts/ner_audit_zs.py
#   Qwen3.5-9B zero-shot    -> venv_train, scripts/ner2_predict_llm.py (as in v1)
# Output: $OUT/<model>.jsonl.gz, one record per passage with LOC spans (word spans s/e, character spans cs/ce, text, conf).
set -u
PASSAGES="${PASSAGES:?}"
OUT="${OUT:?}"
mkdir -p "$OUT"
EN="$OUT/passages_en.jsonl"
python3 - "$PASSAGES" "$EN" <<'PY'
import json, sys
with open(sys.argv[2], 'w', encoding='utf-8') as f:
    for l in open(sys.argv[1], encoding='utf-8'):
        if l.strip() and json.loads(l)['lang'] == 'en':
            f.write(l)
PY
rc=0
source "$ND/venv_train/bin/activate"
for M in r3_mdeberta_HM r3_mmbert_base_HM r3_xlmr_large_H r3_hmteams_HM; do
  echo "=== $M"
  python scripts/ner_audit_predict.py --model-dir "$ND/models/$M" --data "$PASSAGES" --out "$OUT/${M}.jsonl.gz" --words-from tokens || rc=1
done
deactivate
source "$ND/venv_flair/bin/activate"
echo "=== flair ner-english-large (English passages)"
python scripts/ner_audit_zs.py --backend flair --model flair/ner-english-large --map loc=LOC,per=PER --data "$EN" --out "$OUT/flair_ner_english_large.jsonl.gz" || rc=1
deactivate
source "$ND/venv_train/bin/activate"
echo "=== Qwen3.5-9B zero-shot"
python scripts/ner2_predict_llm.py --model Qwen/Qwen3.5-9B --data "$PASSAGES" --out "$OUT/llm_qwen35_9b.jsonl.gz" --answers "$OUT/llm_qwen35_9b_answers.jsonl" || rc=1
exit $rc
