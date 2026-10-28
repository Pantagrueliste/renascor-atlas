#!/bin/bash
# RenNER rescore under the owner rules (decisions of 2026-10-25, guide v0.2), on the Mac (nice 19), with predictions already on disk.
#   dev3   : ner3_eval_v2_dev3.py on gold_dev3.jsonl (v0.1, for the before/after) and gold_dev3_v02.jsonl (owner rules)
#   bench  : ner_editor_bench_score.py --owner-rules (whole bench; teacher and plain LLMs on the 2,000-passage teacher sample)
# Gold v2 and gold v3 are never read. Outputs: v2/renner/rescore_conventions/scores/.
#   bash scripts/ner3_rescore_owner_mac.sh dev3|bench_students|bench_ext|bench_sample
set -euo pipefail
R=/Users/clem/Projects/Papers/renascor-atlas; V=/Users/clem/Projects/Papers/RQ/derived/atlas/ner/v2
O=$V/renner/rescore_conventions; SC=$O/scores; B=$V/editor_bench; PY="nice -n 19 $R/.venv/bin/python"
S=$V/renner/train_v2/seeds; E=$V/external_comparison/pred; L=$V/renner/train_v21/verse_routing; T=$O/pred_teacher_dev3
mkdir -p $SC
p() {  # set model seed -> prediction path
  local set=$1 m=$2 k=$3
  case "$m:$k" in
    v2_xlmrxl_Sfull:1) [ $set = bench ] && echo $B/pred/v2_xlmrxl_Sfull.jsonl.gz || echo $V/renner/train_v2/eval/pred_$set/v2_xlmrxl_Sfull.jsonl.gz ;;
    v2_mdeberta_Sfull:1) [ $set = bench ] && echo $B/pred/v2_mdeberta_Sfull.jsonl.gz || echo $V/renner/train_v2/eval/pred_$set/v2_mdeberta_Sfull.jsonl.gz ;;
    r3_mdeberta_HM:1) [ $set = bench ] && echo $B/pred/r3_mdeberta_HM_gpu.jsonl.gz || echo $V/external_comparison/pred/dev3_ours/r3_mdeberta_HM.jsonl.gz ;;
    v21_mdeberta_lc25:1) echo $V/renner/train_v21/eval/pred_$set/v21_mdeberta_lc25.jsonl.gz ;;
    v21_mdeberta_lc25_lv4:1) echo $L/pred_$set/v21_mdeberta_lc25_lv4.jsonl.gz ;;
    v21_mdeberta_lc25_lv4:*) echo $L/pred_$set/v21_mdeberta_lc25_lv4_s$k.jsonl.gz ;;
    *) echo $S/pred_$set/${m}_s$k.jsonl.gz ;;
  esac
}
EXT="flair_ner_de_large flair_ner_en_large xlmr_conll03 neolatiner davlan_xlmr_large gliner_x_large"
if [ "$1" = dev3 ]; then
  args=(--pred "v2_xlmrxl_Sfull=$(p dev3 v2_xlmrxl_Sfull 1)")
  for k in 1 2 3; do for m in r3_mdeberta_HM v2_mdeberta_Sfull v21_mdeberta_lc25 v21_mdeberta_lc25_lv4; do args+=(--pred "$m@s$k=$(p dev3 $m $k)"); done; done
  for t in silver_v2v3 v3 v4; do args+=(--pred "teacher_$t=$T/teacher_$t.jsonl"); done
  P=$S/plain; for m in gemma4_31b qwen38_27b; do args+=(--pred "${m}_plain=$P/raw/${m}__plain.jsonl" --pred "${m}_plain+prop=$P/${m}__plain+prop.jsonl"); done
  for x in $(ls $E/dev3/*.jsonl.gz); do n=$(basename $x .jsonl.gz); args+=(--pred "$n=$x"); done
  pairs=(--pair "teacher_v4,teacher_v3" --pair "teacher_silver_v2v3,v2_xlmrxl_Sfull" --pair "v2_xlmrxl_Sfull,zs_flair_ner_de_large" --pair "v2_xlmrxl_Sfull,r3_mdeberta_HM@s1")
  for k in 1 2 3; do pairs+=(--pair "v21_mdeberta_lc25_lv4@s$k,v21_mdeberta_lc25@s$k" --pair "v21_mdeberta_lc25@s$k,v2_mdeberta_Sfull@s$k" --pair "v2_mdeberta_Sfull@s$k,r3_mdeberta_HM@s$k"); done
  $PY $R/scripts/ner3_eval_v2_dev3.py --gold $V/renner/dev3/gold_dev3.jsonl "${args[@]}" "${pairs[@]}" --out $SC/dev3_v01.json > $SC/dev3_v01.txt
  $PY $R/scripts/ner3_eval_v2_dev3.py --gold $V/renner/dev3/gold_dev3_v02.jsonl "${args[@]}" "${pairs[@]}" --out $SC/dev3_v02.json > $SC/dev3_v02.txt
fi
if [ "$1" = bench_students ]; then   # whole bench, owner view, every group paired, ref = v2 XL
  args=(--pred "v2_xlmrxl_Sfull=$(p bench v2_xlmrxl_Sfull 1)")
  for k in 1 2 3; do for m in r3_mdeberta_HM v2_mdeberta_Sfull v21_mdeberta_lc25 v21_mdeberta_lc25_lv4; do args+=(--pred "$m@s$k=$(p bench $m $k)"); done; done
  $PY $R/scripts/ner_editor_bench_score.py --bench $B/bench.jsonl --editions $B/editions.json "${args[@]}" --ref v2_xlmrxl_Sfull --owner-rules \
      --out $SC/bench_owner_students.json --md $SC/bench_owner_students.md > $O/logs/bench_owner_students.log 2>&1
fi
if [ "$1" = bench_lv4 ]; then   # lc25-lv4 against lc25 within each seed, owner view, every group paired (as verse_routing, owner gold)
  for k in 1 2 3; do
    $PY $R/scripts/ner_editor_bench_score.py --bench $B/bench.jsonl --editions $B/editions.json --pred "lc25@s$k=$(p bench v21_mdeberta_lc25 $k)" \
        --pred "lc25_lv4@s$k=$(p bench v21_mdeberta_lc25_lv4 $k)" --ref "lc25@s$k" --paired-all-groups --owner-rules \
        --out $SC/bench_owner_lv4_s$k.json --md $SC/bench_owner_lv4_s$k.md > $O/logs/bench_owner_lv4_s$k.log 2>&1
  done
fi
if [ "$1" = bench_ext ]; then
  args=(--pred "v2_xlmrxl_Sfull=$(p bench v2_xlmrxl_Sfull 1)")
  for x in $EXT; do args+=(--pred "zs_$x=$E/bench/zs_$x.jsonl.gz"); done
  $PY $R/scripts/ner_editor_bench_score.py --bench $B/bench.jsonl --editions $B/editions.json "${args[@]}" --ref v2_xlmrxl_Sfull --owner-rules \
      --out $SC/bench_owner_ext.json --md $SC/bench_owner_ext.md > $O/logs/bench_owner_ext.log 2>&1
fi
if [ "$1" = bench_sample ]; then   # teacher and plain LLMs exist only on the teacher sample (2,000 passages)
  P=$S/plain
  args=(--pred "v2_xlmrxl_Sfull=$B/pred/v2_xlmrxl_Sfull.jsonl.gz" --pred "v2_mdeberta_Sfull=$B/pred/v2_mdeberta_Sfull.jsonl.gz" \
        --pred "r3_mdeberta_HM=$B/pred/r3_mdeberta_HM.jsonl.gz" --pred "teacher_gemma4_31b=$B/pred_teacher/gemma4_31b.jsonl")
  for m in gemma4_31b qwen38_27b; do args+=(--pred "${m}_plain=$P/raw/${m}__plain.jsonl" --pred "${m}_plain+prop=$P/${m}__plain+prop.jsonl"); done
  $PY $R/scripts/ner_editor_bench_score.py --bench $B/bench.jsonl --editions $B/editions.json --subset $B/teacher_sample/ids.txt "${args[@]}" \
      --ref v2_xlmrxl_Sfull --owner-rules --out $SC/bench_owner_sample.json --md $SC/bench_owner_sample.md > $O/logs/bench_owner_sample.log 2>&1
  $PY $R/scripts/ner_editor_bench_score.py --bench $B/bench.jsonl --editions $B/editions.json --subset $B/teacher_sample/ids.txt "${args[@]}" \
      --ref v2_xlmrxl_Sfull --out $SC/bench_v01_sample.json --md $SC/bench_v01_sample.md > $O/logs/bench_v01_sample.log 2>&1
fi
echo "done $1"
