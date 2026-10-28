#!/bin/bash
# RenNER Latin verse routing: score v21_mdeberta_lc25_lv4 seeds 1-3 against v21_mdeberta_lc25 seeds 1-3 on the Mac (nice 19), with the
# scripts and settings of the seeds report: dev3 (ner3_eval_v2_dev3.py), editor bench (ner_editor_bench_score.py, every group paired,
# per edition included), test_v2 (ner3_eval_v2_testv2.py). Pairs within each seed. Gold v2 and gold v3 are never read.
#   bash scripts/ner3_lv4_score_mac.sh
set -euo pipefail
R=/Users/clem/Projects/Papers/renascor-atlas; V=/Users/clem/Projects/Papers/RQ/derived/atlas/ner/v2
O=$V/renner/train_v21/verse_routing; SC=$O/scores; B=$V/editor_bench; PY="nice -n 19 $R/.venv/bin/python"
DEV3=$V/renner/dev3/gold_dev3.jsonl; TV2=$V/test2/test_v2.jsonl
mkdir -p $SC $O/hypatia
rsync -a hypatia:renascor-atlas-data/ner/train_lv4/eval/pred_{dev3,testv2,bench} $O/
rsync -a --include='*/' --include='metrics.json' --include='config.json' --include='perf.json' --exclude='*' hypatia:renascor-atlas-data/ner/train_lv4/runs $O/hypatia/
base() {  # set seed
  if [ "$2" = 1 ]; then echo $V/renner/train_v21/eval/pred_$1/v21_mdeberta_lc25.jsonl.gz; else echo $V/renner/train_v2/seeds/pred_$1/v21_mdeberta_lc25_s$2.jsonl.gz; fi
}
new() {   # set seed
  if [ "$2" = 1 ]; then echo $O/pred_$1/v21_mdeberta_lc25_lv4.jsonl.gz; else echo $O/pred_$1/v21_mdeberta_lc25_lv4_s$2.jsonl.gz; fi
}
args=(); pairs=()
for k in 1 2 3; do
  args+=(--pred "lc25@s$k=$(base dev3 $k)" --pred "lc25_lv4@s$k=$(new dev3 $k)"); pairs+=(--pair "lc25_lv4@s$k,lc25@s$k")
done
$PY $R/scripts/ner3_eval_v2_dev3.py --gold $DEV3 "${args[@]}" "${pairs[@]}" --out $SC/dev3.json | tee $SC/dev3.txt
for k in 1 2 3; do
  $PY $R/scripts/ner3_eval_v2_testv2.py --gold $TV2 --pred "lc25@s$k=$(base testv2 $k)" --pred "lc25_lv4@s$k=$(new testv2 $k)" --ref "lc25@s$k" \
      --out $SC/testv2_s$k.json > $SC/testv2_s$k.txt
  $PY $R/scripts/ner_editor_bench_score.py --bench $B/bench.jsonl --editions $B/editions.json --pred "lc25@s$k=$(base bench $k)" \
      --pred "lc25_lv4@s$k=$(new bench $k)" --ref "lc25@s$k" --paired-all-groups --out $SC/bench_s$k.json --md $SC/bench_s$k.md > $SC/bench_s$k.log 2>&1
done
echo done
