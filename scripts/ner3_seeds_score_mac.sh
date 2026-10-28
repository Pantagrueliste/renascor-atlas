#!/bin/bash
# RenNER seeds 1-3 and plain-LLM baseline: scoring on the Mac (nice 19). Copies the predictions from Hypatia, then scores
#   seeds: dev3 (ner3_eval_v2_dev3.py), the editor bench, main editions (ner_editor_bench_score.py), test_v2 (ner3_eval_v2_testv2.py);
#          paired differences within each seed (v2 / v2.1 against the v1 recipe, same bootstrap draws).
#   plain: dev3, and the 2,000-passage teacher sample of the bench (--subset teacher_sample/ids.txt), raw and with name propagation.
# Gold v2 and gold v3 are never read. Outputs: v2/renner/train_v2/seeds/ (scores/, pred_*/).
#   bash scripts/ner3_seeds_score_mac.sh seeds|plain
set -euo pipefail
R=/Users/clem/Projects/Papers/renascor-atlas; N=/Users/clem/Projects/Papers/RQ/derived/atlas/ner; V=$N/v2
S=$V/renner/train_v2/seeds; SC=$S/scores; B=$V/editor_bench; PY="nice -n 19 $R/.venv/bin/python"
DEV3=$V/renner/dev3/gold_dev3.jsonl; TV2=$V/test2/test_v2.jsonl
mkdir -p $SC $S/pred_dev3 $S/pred_testv2 $S/pred_bench $S/plain
# seed-1 predictions (existing files) and seeds 2-3 (this run)
p1() {  # set model -> path of the seed-1 prediction
  case "$1:$2" in
    dev3:v2_mdeberta_Sfull) echo $V/renner/train_v2/eval/pred_dev3/v2_mdeberta_Sfull.jsonl.gz ;;
    testv2:v2_mdeberta_Sfull) echo $V/renner/train_v2/eval/pred_testv2/v2_mdeberta_Sfull.jsonl.gz ;;
    bench:v2_mdeberta_Sfull) echo $B/pred/v2_mdeberta_Sfull.jsonl.gz ;;
    dev3:v21_mdeberta_lc25|testv2:v21_mdeberta_lc25|bench:v21_mdeberta_lc25) echo $V/renner/train_v21/eval/pred_$1/v21_mdeberta_lc25.jsonl.gz ;;
    dev3:r3_mdeberta_HM) echo $V/external_comparison/pred/dev3_ours/r3_mdeberta_HM.jsonl.gz ;;
    testv2:r3_mdeberta_HM) echo $V/renner/train_v2/eval/pred_testv2/r3_mdeberta_HM.jsonl.gz ;;
    bench:r3_mdeberta_HM) echo $B/pred/r3_mdeberta_HM_gpu.jsonl.gz ;;   # GPU run, like seeds 2-3
  esac
}
pp() { # set model seed
  if [ "$3" = 1 ]; then p1 "$1" "$2"; else echo $S/pred_$1/$2_s$3.jsonl.gz; fi
}
if [ "$1" = seeds ]; then
  rsync -a hypatia:renascor-atlas-data/ner/train_seeds/eval/pred_{dev3,testv2,bench} $S/
  rsync -a --include='*/' --include='metrics.json' --include='config.json' --include='perf.json' --exclude='*' hypatia:renascor-atlas-data/ner/train_seeds/runs $S/hypatia/
  args=(); pairs=()
  for k in 1 2 3; do
    for m in v2_mdeberta_Sfull v21_mdeberta_lc25 r3_mdeberta_HM; do args+=(--pred "$m@s$k=$(pp dev3 $m $k)"); done
    pairs+=(--pair "v2_mdeberta_Sfull@s$k,r3_mdeberta_HM@s$k" --pair "v21_mdeberta_lc25@s$k,r3_mdeberta_HM@s$k" --pair "v21_mdeberta_lc25@s$k,v2_mdeberta_Sfull@s$k")
  done
  $PY $R/scripts/ner3_eval_v2_dev3.py --gold $DEV3 "${args[@]}" "${pairs[@]}" --out $SC/dev3.json | tee $SC/dev3.txt
  for k in 1 2 3; do
    args=(); for m in r3_mdeberta_HM v2_mdeberta_Sfull v21_mdeberta_lc25; do args+=(--pred "$m@s$k=$(pp testv2 $m $k)"); done
    $PY $R/scripts/ner3_eval_v2_testv2.py --gold $TV2 "${args[@]}" --ref "r3_mdeberta_HM@s$k" --out $SC/testv2_s$k.json > $SC/testv2_s$k.txt
    args=(); for m in r3_mdeberta_HM v2_mdeberta_Sfull v21_mdeberta_lc25; do args+=(--pred "$m@s$k=$(pp bench $m $k)"); done
    $PY $R/scripts/ner_editor_bench_score.py --bench $B/bench.jsonl --editions $B/editions.json "${args[@]}" --ref "r3_mdeberta_HM@s$k" \
        --out $SC/bench_main_s$k.json --md $SC/bench_main_s$k.md > $SC/bench_main_s$k.log 2>&1
    # v2.1 against v2 within the seed
    args=(); for m in v2_mdeberta_Sfull v21_mdeberta_lc25; do args+=(--pred "$m@s$k=$(pp testv2 $m $k)"); done
    $PY $R/scripts/ner3_eval_v2_testv2.py --gold $TV2 "${args[@]}" --ref "v2_mdeberta_Sfull@s$k" --out $SC/testv2_lc_s$k.json > $SC/testv2_lc_s$k.txt
    args=(); for m in v2_mdeberta_Sfull v21_mdeberta_lc25; do args+=(--pred "$m@s$k=$(pp bench $m $k)"); done
    $PY $R/scripts/ner_editor_bench_score.py --bench $B/bench.jsonl --editions $B/editions.json "${args[@]}" --ref "v2_mdeberta_Sfull@s$k" \
        --out $SC/bench_lc_s$k.json > $SC/bench_lc_s$k.log 2>&1
  done
fi
if [ "$1" = plain ]; then
  P=$S/plain; rsync -a hypatia:renascor-atlas-data/ner/plain_llm/pred/ $P/raw/
  rsync -a hypatia:renascor-atlas-data/ner/plain_llm/logs/ $P/logs/
  # name propagation (every occurrence of a predicted string gets its label), the only post-processing of the plain variant "+prop"
  for m in gemma4_31b qwen38_27b; do
    $PY $R/scripts/ner3_plain_llm_prop.py --passages $V/renner/dev3/passages.jsonl --passages $B/teacher_sample/passages_la.jsonl \
        --passages $B/teacher_sample/passages_other.jsonl --pred $P/raw/${m}__plain.jsonl --out $P/${m}__plain+prop.jsonl
  done
  T=$V/renner/silver/dev3_check_hypatia/pred
  args=(--pred "v2_xlmrxl_Sfull=$V/renner/train_v2/eval/pred_dev3/v2_xlmrxl_Sfull.jsonl.gz" --pred "v2_mdeberta_Sfull=$(p1 dev3 v2_mdeberta_Sfull)" --pred "r3_mdeberta_HM=$(p1 dev3 r3_mdeberta_HM)")
  for m in gemma4_31b qwen38_27b; do args+=(--pred "${m}_plain=$P/raw/${m}__plain.jsonl" --pred "${m}_plain+prop=$P/${m}__plain+prop.jsonl"); done
  pairs=(); for m in gemma4_31b_plain gemma4_31b_plain+prop qwen38_27b_plain qwen38_27b_plain+prop; do pairs+=(--pair "$m,r3_mdeberta_HM" --pair "$m,v2_xlmrxl_Sfull"); done
  $PY $R/scripts/ner3_eval_v2_dev3.py --gold $DEV3 "${args[@]}" --teacher es,it=$T/gemma4_31b__fewshotv2.jsonl --teacher la=$T/gemma4_31b__fewshotv3.jsonl \
      "${pairs[@]}" --pair "gemma4_31b_plain,teacher_gemma4_31b_pp" --out $SC/plain_dev3.json | tee $SC/plain_dev3.txt
  for ref in v2_xlmrxl_Sfull gemma4_31b_plain; do
    args=(--pred "v2_xlmrxl_Sfull=$B/pred/v2_xlmrxl_Sfull.jsonl.gz" --pred "v2_mdeberta_Sfull=$B/pred/v2_mdeberta_Sfull.jsonl.gz" --pred "r3_mdeberta_HM=$B/pred/r3_mdeberta_HM.jsonl.gz" \
          --pred "teacher_gemma4_31b=$B/pred_teacher/gemma4_31b.jsonl")
    for m in gemma4_31b qwen38_27b; do args+=(--pred "${m}_plain=$P/raw/${m}__plain.jsonl" --pred "${m}_plain+prop=$P/${m}__plain+prop.jsonl"); done
    $PY $R/scripts/ner_editor_bench_score.py --bench $B/bench.jsonl --editions $B/editions.json --subset $B/teacher_sample/ids.txt "${args[@]}" --ref $ref \
        --out $SC/plain_bench_sample_ref_$ref.json --md $SC/plain_bench_sample_ref_$ref.md > $SC/plain_bench_sample_ref_$ref.log 2>&1
  done
fi
echo "done $1"
