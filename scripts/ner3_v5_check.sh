#!/bin/bash
# RenNER teacher prompt v5 (owner rules of 2026-10-25) on dev3 (+ the 40 Latin audit passages, as silver/v4_check), from the Mac.
#   bash scripts/ner3_v5_check.sh submit   # copies the two prompts, counts our GPU jobs (max 2, the XL chain counts), submits one job
#   bash scripts/ner3_v5_check.sh score    # copies the predictions back and scores them on the Mac (nice 19)
# Rule: v2/renner/rescore_conventions/regle_prompt_v5.md. Gold v2 and gold v3 are never read. gpu03 excluded.
set -euo pipefail
R=/Users/clem/Projects/Papers/renascor-atlas; V=/Users/clem/Projects/Papers/RQ/derived/atlas/ner/v2; O=$V/renner/rescore_conventions
PY="nice -n 19 $R/.venv/bin/python"
if [ "$1" = submit ]; then
  scp -q $R/scripts/renner_prompt_rules_en_v5.md $R/scripts/renner_prompt_rules_en_v5v.md hypatia:renascor-atlas/scripts/
  ssh hypatia 'cd renascor-atlas && md5sum scripts/renner_prompt_rules_en_v5.md scripts/renner_prompt_rules_en_v5v.md scripts/renner_fewshot_v4.jsonl scripts/renner_fewshot_v1train.jsonl'
  md5 $R/scripts/renner_prompt_rules_en_v5.md $R/scripts/renner_prompt_rules_en_v5v.md $R/scripts/renner_fewshot_v4.jsonl $R/scripts/renner_fewshot_v1train.jsonl
  n=$(ssh hypatia 'squeue -h -u $USER -o "%P" | grep -ci gpu' || true)
  echo "our GPU jobs now: $n"
  if [ "${n:-0}" -ge 2 ]; then echo "2 GPU jobs already: not submitting"; exit 3; fi
  ssh hypatia 'D=$HOME/renascor-atlas-data/ner; mkdir -p $D/silver/v5_check/pred $D/silver/logs; cd $HOME/renascor-atlas && \
    sbatch -J r8-v5check -t 01:00:00 --exclude=gpu03 scripts/ner3_silver_vllm_job.sh \
    "--model google/gemma-4-31B-it --name gemma4_31b --text-only --max-model-len 8192 --prompts fewshot --passages $D/silver/v4_check/in/dev3_plus_audit40.jsonl --outdir $D/silver/v5_check/pred --guide v5=scripts/renner_prompt_rules_en_v5.md --guide v5v=scripts/renner_prompt_rules_en_v5v.md --guide-fewshot v5v=scripts/renner_fewshot_v4.jsonl"'
fi
if [ "$1" = score ]; then
  P=$V/renner/silver/v5_check; mkdir -p $P/pred $P/logs
  rsync -a hypatia:renascor-atlas-data/ner/silver/v5_check/pred/ $P/pred/
  rsync -a --include='r8-v5check-*' --exclude='*' hypatia:renascor-atlas-data/ner/silver/logs/ $P/logs/
  T=$O/pred_teacher_dev3; DP=$V/renner/dev3/passages.jsonl
  $PY $R/scripts/ner3_rescore_teacher_pp.py --passages $DP --out $T/teacher_v5.jsonl es,it,la=$P/pred/gemma4_31b__fewshotv5.jsonl
  $PY $R/scripts/ner3_rescore_teacher_pp.py --passages $DP --out $T/teacher_v5v.jsonl es,it,la=$P/pred/gemma4_31b__fewshotv5v.jsonl
  args=(); for t in silver_v2v3 v3 v4 v5 v5v; do args+=(--pred "teacher_$t=$T/teacher_$t.jsonl"); done
  $PY $R/scripts/ner3_eval_v2_dev3.py --gold $V/renner/dev3/gold_dev3_v02.jsonl "${args[@]}" --pair teacher_v5,teacher_silver_v2v3 --pair teacher_v5,teacher_v3 \
      --pair teacher_v5v,teacher_v4 --out $O/scores/dev3_v02_v5.json | tee $O/scores/dev3_v02_v5.txt
  $PY $R/scripts/ner3_dev3_rule_violations.py --v01 $V/renner/dev3/gold_dev3.jsonl --v02 $V/renner/dev3/gold_dev3_v02.jsonl \
      --pred "teacher_silver=$T/teacher_silver_v2v3.jsonl" --pred "teacher_v4=$T/teacher_v4.jsonl" --pred "teacher_v5=$T/teacher_v5.jsonl" \
      --pred "teacher_v5v=$T/teacher_v5v.jsonl" --out $O/conformite/dev3_violations_v5.json | tee $O/conformite/dev3_violations_v5.txt
fi
