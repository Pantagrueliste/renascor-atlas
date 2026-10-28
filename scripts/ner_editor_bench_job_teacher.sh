#!/bin/bash
# Editor-tag bench, teacher on the stratified 2,000-passage sample (scripts/ner_editor_bench_sample.py): google/gemma-4-31B-it with vLLM,
# fewshot prompt, guide v3 for Latin and v2 for the other languages (as the silver runs, scripts/ner3_silver_vllm_job.sh). One GPU, <= 1 h.
#   sbatch -J eb-teacher scripts/ner_editor_bench_job_teacher.sh
# Output: $E/pred_teacher/gemma4_31b__{la,other}.jsonl (resumable: ids already written are skipped).
#SBATCH -p gpu.A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=200G
#SBATCH -t 01:00:00
#SBATCH -o /home/cag30/renascor-atlas-data/ner/logs/%x-%j.out
set -uo pipefail
D="$HOME/renascor-atlas-data/ner"; E="$D/v2/editor_bench"; S="$E/teacher_sample"; O="$E/pred_teacher"; mkdir -p "$O"
COMMON="--model google/gemma-4-31B-it --name gemma4_31b --text-only --max-model-len 8192 --prompts fewshot --batch 500"
bash "$HOME/renascor-atlas/scripts/ner3_silver_vllm_job.sh" \
  "$COMMON --guide scripts/renner_prompt_rules_en_v3.md --passages $S/passages_la.jsonl --out $O/gemma4_31b__la.jsonl" \
  "$COMMON --guide scripts/renner_prompt_rules_en_v2.md --passages $S/passages_other.jsonl --out $O/gemma4_31b__other.jsonl"
