#!/bin/bash
# RenNER v2.1 scoring on the Mac (nice 19): copy the predictions from Hypatia, apply the frozen post-processing, score dev3, bench-dev,
# the editor bench (main editions: scores only) and test_v2. Gold v2 and gold v3 are never read.
#   [TAG=xl] bash scripts/ner3_v21_score_mac.sh STAGE REF_MODEL [MODEL ...]   (STAGE = dev: dev3 + bench-dev; final: + bench main + test_v2;
#   the first model is the reference of the paired differences; outputs in train_v21/eval/scores[_TAG]/)
set -euo pipefail
STAGE="$1"; shift
R=/Users/clem/Projects/Papers/renascor-atlas; N=/Users/clem/Projects/Papers/RQ/derived/atlas/ner; T=$N/v2/renner/train_v21; E=$T/eval
B=$N/v2/editor_bench; PY="nice -n 19 $R/.venv/bin/python"
RULES=$(python3 -c "import json; print(','.join(json.load(open('$T/pp/pp_rules_frozen.json'))['rules']))")
SC=$E/scores${TAG:+_$TAG}
mkdir -p $E/{pred_dev3,pred_testv2,pred_bench,pred_bench_dev_lc} $SC
rsync -a hypatia:renascor-atlas-data/ner/train_v21/eval/pred_{dev3,testv2,bench,bench_dev_lc} $E/
for M in "$@"; do
  for s in dev3 bench bench_dev_lc testv2; do
    case $s in dev3) DATA=$N/v2/renner/dev3/gold_dev3.jsonl ;; bench) DATA=$B/bench.jsonl ;;
               bench_dev_lc) DATA=$T/bench_dev/bench_dev_lc.jsonl ;; testv2) DATA=$N/v2/test2/test_v2.jsonl ;; esac
    IN=$E/pred_$s/$M.jsonl.gz; OUT=$E/pred_$s/$M+pp.jsonl.gz
    [ -s "$IN" ] && { [ -s "$OUT" ] || $PY $R/scripts/ner3_v21_apply_pp.py --data $DATA --pred $IN --rules $RULES --out $OUT; }
  done
done
ALL=(); for M in "$@"; do ALL+=("$M" "$M+pp"); done
args=(); for M in "${ALL[@]}"; do args+=(--pred "$M=$E/pred_dev3/$M.jsonl.gz"); done
pairs=(); for M in "${ALL[@]}"; do [ "$M" != "$1" ] && pairs+=(--pair "$M,$1"); done
$PY $R/scripts/ner3_eval_v2_dev3.py --gold $N/v2/renner/dev3/gold_dev3.jsonl "${args[@]}" "${pairs[@]}" --out $SC/dev3.json | tee $SC/dev3.txt
args=(); for M in "${ALL[@]}"; do args+=(--pred "$M=$E/pred_bench/$M.jsonl.gz"); [ -s $E/pred_bench_dev_lc/$M.jsonl.gz ] && args+=(--pred-lc "$M=$E/pred_bench_dev_lc/$M.jsonl.gz"); done
$PY $R/scripts/ner3_v21_bench_dev.py score --dir $T/bench_dev --editions $B/editions.json "${args[@]}" --ref "$1" --out $SC/bench_dev.json --md $SC/bench_dev.md
if [ "$STAGE" = final ]; then
  args=(); for M in "${ALL[@]}"; do args+=(--pred "$M=$E/pred_bench/$M.jsonl.gz"); done
  $PY $R/scripts/ner_editor_bench_score.py --bench $B/bench.jsonl --editions $B/editions.json "${args[@]}" --ref "$1" \
      --out $SC/bench_main.json --md $SC/bench_main.md > $SC/bench_main.log 2>&1
  args=(); for M in "${ALL[@]}"; do args+=(--pred "$M=$E/pred_testv2/$M.jsonl.gz"); done
  $PY $R/scripts/ner3_eval_v2_testv2.py --gold $N/v2/test2/test_v2.jsonl "${args[@]}" --ref "$1" --out $SC/testv2.json | tee $SC/testv2.txt
fi
