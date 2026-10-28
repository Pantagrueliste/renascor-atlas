#!/bin/bash
# NER v2, large encoders: tokenise the HM mix of v1 and the v1 dev set into the shared cache (CPU, debug partition), and count the training windows
# of a silver sample. Run from ~/renascor-atlas inside venv_train (slurm/ner_audit_cpu.sbatch-like environment). Gold v2 is never read.
set -u
source scripts/ner_sets.sh
source scripts/ner_mixes.sh
S="$ND/v3/enc_smoke"
mkdir -p "$S/runs"
for spec in "xlxl|facebook/xlm-roberta-xl" "eb2b|$ND/local_models/EuroBERT-2.1B"; do
  IFS='|' read -r n MP <<< "$spec"
  for ld in 300 1000; do
    python -m ratlas.ner.train --name "enc_$n" --model "$MP" $MIX_HM --dev "$OURS/dev.jsonl" --out "$S/runs" --cache "$ND/cache" \
        --max-len 256 --seed 1 --limit-dev $ld --encode-only
    echo "[encode $n dev$ld] exit $?"
  done
done
python scripts/ner3_enc_windows.py --sample "$S/silver_sample.jsonl" --tok "xlmr=facebook/xlm-roberta-xl" \
    --tok "eurobert=$ND/local_models/EuroBERT-2.1B" --force-bos eurobert --out "$S/silver_windows.json"
echo "[windows] exit $?"
